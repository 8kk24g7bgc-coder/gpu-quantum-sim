"""Publication-quality plots for the benchmark and VQE results.

Plots are built only from the CSV files the experiments write, so they can be
regenerated at any time without re-running a simulation::

    python -m src.plotting ghz results/ghz_benchmark.csv results/ghz_benchmark_cpu_n24.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # headless: works on cluster nodes and in CI
import matplotlib.pyplot as plt  # noqa: E402

NVIDIA_GREEN = "#76B900"
CPU_COLORS = ["#4C72B0", "#8172B2", "#64B5CD", "#937860"]
GPU_COLORS = [NVIDIA_GREEN, "#2E7D32", "#9CCC65"]

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
}

Series = dict[str, list[tuple[int, float, float]]]  # label -> [(n, median, std)]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_ghz_rows(paths: Iterable[Path]) -> list[dict]:
    """Read and concatenate GHZ benchmark CSVs."""
    rows: list[dict] = []
    for path in paths:
        with Path(path).open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _series_label(row: dict) -> str:
    kind = "GPU" if row["device"] == "gpu" else "CPU"
    return f"{kind} · {row['target']} ({row['precision']}) · {row['device_name']}"


def ghz_series(rows: list[dict]) -> tuple[Series, Series]:
    """Split successful rows into CPU and GPU series keyed by a readable label."""
    cpu: Series = defaultdict(list)
    gpu: Series = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok" or not row.get("median_s"):
            continue
        point = (int(row["n"]), float(row["median_s"]), float(row["std_s"] or 0.0))
        (gpu if row["device"] == "gpu" else cpu)[_series_label(row)].append(point)
    for series in (cpu, gpu):
        for points in series.values():
            points.sort()
    return dict(cpu), dict(gpu)


# --------------------------------------------------------------------------- #
# GHZ plots
# --------------------------------------------------------------------------- #
def plot_ghz_runtime(rows: list[dict], out_path: Path) -> Path:
    """Runtime vs number of qubits (log y), one line per target/device."""
    cpu, gpu = ghz_series(rows)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for colors, series, marker in ((CPU_COLORS, cpu, "o"), (GPU_COLORS, gpu, "s")):
            for color, (label, points) in zip(colors, series.items()):
                n, med, std = zip(*points)
                ax.errorbar(n, med, yerr=std, marker=marker, ms=5, lw=1.8, capsize=3,
                            color=color, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Number of qubits $n$")
        ax.set_ylabel("Wall-clock time per run [s] (median)")
        ax.set_title("GHZ statevector simulation: CPU vs GPU")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=9, loc="upper left")
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def speedup_curves(rows: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """CPU-median / GPU-median at every n measured on both, per (CPU, GPU) pair."""
    cpu, gpu = ghz_series(rows)
    curves: dict[str, list[tuple[int, float]]] = {}
    for cpu_label, cpu_points in cpu.items():
        cpu_by_n = {n: med for n, med, _ in cpu_points}
        for gpu_label, gpu_points in gpu.items():
            pairs = [(n, cpu_by_n[n] / med) for n, med, _ in gpu_points if n in cpu_by_n and med > 0]
            if pairs:
                cpu_name = cpu_label.split(" · ")[-1]
                gpu_name = gpu_label.split(" · ")[-1]
                curves[f"{gpu_name} vs {cpu_name}"] = sorted(pairs)
    return curves


def plot_ghz_speedup(rows: list[dict], out_path: Path) -> Optional[Path]:
    """GPU speedup (CPU time / GPU time) vs qubits; ``None`` if no GPU data."""
    curves = speedup_curves(rows)
    if not curves:
        return None
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for color, (label, points) in zip(GPU_COLORS + CPU_COLORS, curves.items()):
            n, s = zip(*points)
            ax.plot(n, s, marker="o", lw=2, color=color, label=label)
            ax.annotate(f"{s[-1]:.0f}×", (n[-1], s[-1]), textcoords="offset points",
                        xytext=(6, 4), fontsize=10, color=color, fontweight="bold")
        ax.axhline(1.0, color="0.5", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("Number of qubits $n$")
        ax.set_ylabel("Speedup (CPU time / GPU time)")
        ax.set_title("GPU speedup on GHZ circuits")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=9, loc="upper left")
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_ghz_results(csv_paths: Iterable[Path], out_dir: Path) -> list[Path]:
    """Write ``ghz_benchmark.png`` and, if GPU data exist, ``ghz_speedup.png``."""
    rows = load_ghz_rows(csv_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [plot_ghz_runtime(rows, out_dir / "ghz_benchmark.png")]
    speedup = plot_ghz_speedup(rows, out_dir / "ghz_speedup.png")
    if speedup is None:
        print("No GPU results found: ghz_speedup.png not written.")
    else:
        written.append(speedup)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate plots from result CSVs.")
    sub = parser.add_subparsers(dest="kind", required=True)
    ghz_parser = sub.add_parser("ghz", help="GHZ benchmark plots")
    ghz_parser.add_argument("csv", nargs="+", type=Path)
    ghz_parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    if args.kind == "ghz":
        for path in plot_ghz_results(args.csv, args.outdir):
            print(f"Plot written to {path}")


if __name__ == "__main__":
    main()
