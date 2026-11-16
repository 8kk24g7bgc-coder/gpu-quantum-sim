"""GHZ statevector benchmark: CPU (``qpp-cpu``) vs GPU (``nvidia``, cuStateVec).

The GHZ circuit puts ``n`` qubits into (|0...0> + |1...1>)/sqrt(2) with one
Hadamard and a chain of ``n - 1`` CNOTs. It is a clean scaling test: every
gate acts on a statevector of ``2**n`` amplitudes, so runtime and memory both
grow exponentially with ``n``.

For each target and each qubit count the script runs one warm-up and then
``--repeats`` timed calls to ``cudaq.sample`` and records the median and the
standard deviation of the wall-clock time. Sizes that would not fit in RAM or
VRAM are skipped (and recorded as skipped) instead of crashing.

Usage (from the repository root)::

    python -m src.ghz_benchmark                 # full run: n = 20..30
    python -m src.ghz_benchmark --quick         # < 2 minutes on a CPU
    python -m src.ghz_benchmark --targets qpp-cpu --max-qubits 32 \\
        --output results/ghz_benchmark_cpu_n24.csv      # cluster CPU run
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# CUDA-Q 0.16 prints two notices on import that do not concern this project:
# the preview status of the optional `cudaq-logical` package and a heads-up
# about future API changes (we pin cudaq==0.16.0.post1 for exactly that
# reason). Silence them so the benchmark log stays readable.
warnings.filterwarnings("ignore", message=r"cudaq-logical is in preview", category=UserWarning)
warnings.filterwarnings("ignore", message=r"The CUDA-Q `sample` and `observe`", category=FutureWarning)

import cudaq  # noqa: E402

from src.hardware import (  # noqa: E402
    available_gpu_memory_bytes,
    available_host_memory_bytes,
    collect_hardware_info,
    estimate_state_memory_gb,
    fits_in_memory,
    gpu_available,
    query_gpus,
)

CSV_FIELDS: list[str] = [
    "target",
    "n",
    "median_s",
    "std_s",
    "est_mem_gb",
    "precision",
    "device",
    "device_name",
    "host",
    "shots",
    "repeats",
    "status",
]


# --------------------------------------------------------------------------- #
# Kernel and correctness check
# --------------------------------------------------------------------------- #
@cudaq.kernel
def ghz(n: int):
    """Prepare an n-qubit GHZ state and measure every qubit."""
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    mz(q)


def sample_ghz(n: int, shots: int) -> dict[str, int]:
    """Sample the GHZ kernel on the current target and return bitstring counts."""
    return dict(cudaq.sample(ghz, n, shots_count=shots).items())


def ghz_counts_are_valid(counts: dict[str, int], n: int, tolerance: float = 0.05) -> bool:
    """Check that GHZ sampling produced only |0...0> and |1...1>, each near 50%.

    Args:
        counts: Bitstring -> number of shots.
        n: Number of qubits.
        tolerance: Allowed deviation of each outcome's frequency from 0.5.
    """
    total = sum(counts.values())
    zeros, ones = "0" * n, "1" * n
    if total == 0 or set(counts) - {zeros, ones}:
        return False
    return all(abs(counts.get(key, 0) / total - 0.5) <= tolerance for key in (zeros, ones))


def check_ghz_correctness(n: int = 8, shots: int = 4000, tolerance: float = 0.05) -> dict[str, int]:
    """Sample a small GHZ state on the current target and assert it is correct.

    With 4000 shots the standard deviation of each frequency is about 0.008,
    so a tolerance of 0.05 is a ~6-sigma bound: a failure means a real bug.

    Returns:
        The measured counts.

    Raises:
        AssertionError: If any other bitstring appears or the split is off.
    """
    counts = sample_ghz(n, shots)
    assert ghz_counts_are_valid(counts, n, tolerance), f"GHZ({n}) check failed: {counts}"
    return counts


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetSpec:
    """A CUDA-Q simulation target and how it stores amplitudes.

    ``headroom`` is the peak memory of a run divided by the bare statevector
    size; a qubit count is skipped when ``state_size * headroom`` exceeds the
    free memory.
    """

    name: str
    precision: str
    device: str  # "cpu" or "gpu"
    headroom: float
    options: dict = field(default_factory=dict)

    def activate(self) -> None:
        """Make this the active CUDA-Q target."""
        cudaq.set_target(self.name, **self.options)


TARGETS: dict[str, TargetSpec] = {
    # Single-node CPU simulator (Q++), always double precision. Headroom 2.0 is
    # measured, not guessed: on a node with 123 GiB free, n = 31 (32 GiB state)
    # ran but n = 32 (64 GiB state) was killed by the OOM killer, so the peak
    # usage is between 1.9x and 2x the statevector (Slurm job 51371).
    "qpp-cpu": TargetSpec("qpp-cpu", "fp64", "cpu", headroom=2.0),
    # cuStateVec on one GPU; fp32 is the default, set explicitly for clarity.
    "nvidia": TargetSpec("nvidia", "fp32", "gpu", headroom=1.5, options={"option": "fp32"}),
}


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def time_sample(n: int, shots: int, warmup: int, repeats: int) -> list[float]:
    """Run ``warmup`` untimed and ``repeats`` timed GHZ samples; return seconds."""
    for _ in range(warmup):
        cudaq.sample(ghz, n, shots_count=shots)
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        cudaq.sample(ghz, n, shots_count=shots)
        times.append(time.perf_counter() - start)
    return times


def _available_bytes(spec: TargetSpec) -> Optional[int]:
    if spec.device == "cpu":
        return available_host_memory_bytes()
    return available_gpu_memory_bytes()


def _device_name(spec: TargetSpec, cpu_model: str) -> str:
    if spec.device == "cpu":
        return cpu_model
    gpus = query_gpus()
    return gpus[0].name if gpus else "NVIDIA GPU"


def benchmark_target(
    spec: TargetSpec,
    qubit_counts: list[int],
    shots: int,
    warmup: int,
    repeats: int,
    headroom: float,
    host: str,
    cpu_model: str,
    time_budget_s: Optional[float],
    writer: "ResultWriter",
) -> None:
    """Benchmark one target over ``qubit_counts`` and stream rows to ``writer``."""
    device_name = _device_name(spec, cpu_model)
    started = time.perf_counter()
    last_median: Optional[float] = None

    for n in qubit_counts:
        row = {
            "target": spec.name,
            "n": n,
            "median_s": "",
            "std_s": "",
            "est_mem_gb": f"{estimate_state_memory_gb(n, spec.precision):.6g}",
            "precision": spec.precision,
            "device": spec.device,
            "device_name": device_name,
            "host": host,
            "shots": shots,
            "repeats": repeats,
            "status": "ok",
        }

        available = _available_bytes(spec)
        if available is not None and not fits_in_memory(n, spec.precision, available, headroom):
            row["status"] = "skipped_memory"
            print(
                f"  [{spec.name}] n={n:2d}  SKIP: needs {row['est_mem_gb']} GiB x{headroom} "
                f"headroom, only {available / 1024**3:.1f} GiB available"
            )
            writer.write(row)
            continue

        if time_budget_s is not None and last_median is not None:
            # Each extra qubit roughly doubles the cost of every gate.
            projected = 2.2 * last_median * (warmup + repeats)
            if time.perf_counter() - started + projected > time_budget_s:
                row["status"] = "skipped_time"
                print(f"  [{spec.name}] n={n:2d}  SKIP: projected {projected:.0f} s exceeds time budget")
                writer.write(row)
                continue

        try:
            times = time_sample(n, shots, warmup, repeats)
        except (RuntimeError, MemoryError) as exc:
            row["status"] = "error"
            print(f"  [{spec.name}] n={n:2d}  ERROR: {exc}")
            writer.write(row)
            continue

        last_median = statistics.median(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        row["median_s"] = f"{last_median:.6g}"
        row["std_s"] = f"{std:.6g}"
        print(
            f"  [{spec.name}] n={n:2d}  median {last_median:9.4f} s  std {std:8.4f} s  "
            f"state {row['est_mem_gb']} GiB"
        )
        writer.write(row)


class ResultWriter:
    """Append rows to a CSV and flush after each one.

    Writing row by row means a crash or a node failure late in a long run
    still leaves every finished measurement on disk.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--targets", default="qpp-cpu,nvidia",
                        help="comma-separated CUDA-Q targets (default: qpp-cpu,nvidia)")
    parser.add_argument("--min-qubits", type=int, default=20)
    parser.add_argument("--max-qubits", type=int, default=30)
    parser.add_argument("--step", type=int, default=1, help="qubit-count step")
    parser.add_argument("--shots", type=int, default=1000, help="shots per cudaq.sample call")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per size")
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per size")
    parser.add_argument("--headroom", type=float, default=None,
                        help="override the per-target memory headroom "
                             "(skip n if state size x headroom exceeds free memory)")
    parser.add_argument("--time-budget", type=float, default=None,
                        help="optional seconds per target; skip sizes projected to exceed it")
    parser.add_argument("--output", type=Path, default=Path("results/ghz_benchmark.csv"))
    parser.add_argument("--no-plot", action="store_true", help="do not write the PNG plots")
    parser.add_argument("--quick", action="store_true",
                        help="reduced run (n = 10..20 step 2, 3 repeats, 100 shots)")
    args = parser.parse_args(argv)

    if args.quick:
        args.min_qubits, args.max_qubits, args.step = 10, 20, 2
        args.repeats, args.shots = 3, 100
    if args.min_qubits < 2 or args.max_qubits < args.min_qubits:
        parser.error("need 2 <= --min-qubits <= --max-qubits")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    hw = collect_hardware_info()
    print("Hardware:", json.dumps(hw.to_dict(), indent=2))

    hw_path = args.output.with_name(args.output.stem + "_hardware.json")
    hw_path.parent.mkdir(parents=True, exist_ok=True)
    hw_path.write_text(json.dumps(hw.to_dict(), indent=2) + "\n")

    requested = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TARGETS]
    if unknown:
        print(f"Unknown target(s) {unknown}; choose from {sorted(TARGETS)}", file=sys.stderr)
        return 2

    qubit_counts = list(range(args.min_qubits, args.max_qubits + 1, args.step))
    writer = ResultWriter(args.output)
    ran_any = False
    try:
        for name in requested:
            spec = TARGETS[name]
            if spec.device == "gpu" and not gpu_available():
                print(f"\n[{name}] No NVIDIA GPU available: skipping the GPU benchmark.")
                continue

            headroom = args.headroom if args.headroom is not None else spec.headroom
            print(f"\n[{name}] precision={spec.precision}  shots={args.shots}  "
                  f"warmup={args.warmup}  repeats={args.repeats}  headroom={headroom}")
            spec.activate()
            counts = check_ghz_correctness()
            print(f"  correctness check GHZ(8): {counts}  -> OK")

            benchmark_target(spec, qubit_counts, args.shots, args.warmup, args.repeats,
                             headroom, hw.hostname, hw.cpu_model, args.time_budget, writer)
            ran_any = True
    finally:
        writer.close()

    print(f"\nResults written to {args.output}")
    if ran_any and not args.no_plot:
        from src.plotting import plot_ghz_results  # noqa: PLC0415

        for path in plot_ghz_results([args.output], args.output.parent):
            print(f"Plot written to {path}")
    return 0 if ran_any else 1


if __name__ == "__main__":
    sys.exit(main())
