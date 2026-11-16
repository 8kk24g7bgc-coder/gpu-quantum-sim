"""Hardware detection and memory estimation for statevector simulation.

Everything here is plain Python (no psutil), so it works the same on an HPC
node, on Google Colab and on a laptop. CUDA-Q is imported lazily, which means
this module can be imported and unit-tested without a working CUDA-Q install.

Memory model
------------
A statevector of ``n`` qubits holds ``2**n`` complex amplitudes:

* fp32 (complex64)  -> 8 bytes per amplitude
* fp64 (complex128) -> 16 bytes per amplitude

Simulators also need some workspace on top of the state itself (for example
buffers used while sampling), so callers decide whether a size fits with a
``headroom`` multiplier rather than with the bare state size.

Run ``python -m src.hardware`` to print a JSON summary of the machine.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

BYTES_PER_AMPLITUDE: dict[str, int] = {"fp32": 8, "fp64": 16}
GIB: int = 1024**3

# cgroup v1 reports "no limit" as a huge number close to 2**63.
_CGROUP_V1_UNLIMITED = 1 << 60


# --------------------------------------------------------------------------- #
# Memory estimator
# --------------------------------------------------------------------------- #
def bytes_per_amplitude(precision: str) -> int:
    """Return the size in bytes of one complex amplitude.

    Args:
        precision: ``"fp32"`` (complex64) or ``"fp64"`` (complex128).

    Raises:
        ValueError: If ``precision`` is not recognised.
    """
    try:
        return BYTES_PER_AMPLITUDE[precision]
    except KeyError:
        raise ValueError(
            f"precision must be one of {sorted(BYTES_PER_AMPLITUDE)}, got {precision!r}"
        ) from None


def estimate_state_memory_bytes(n_qubits: int, precision: str) -> int:
    """Return the exact size in bytes of an ``n_qubits`` statevector.

    Args:
        n_qubits: Number of qubits (must be >= 1).
        precision: ``"fp32"`` or ``"fp64"``.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    return (1 << n_qubits) * bytes_per_amplitude(precision)


def estimate_state_memory_gb(n_qubits: int, precision: str) -> float:
    """Statevector size in GiB (2**30 bytes), as written to the results CSV."""
    return estimate_state_memory_bytes(n_qubits, precision) / GIB


def fits_in_memory(
    n_qubits: int,
    precision: str,
    available_bytes: int,
    headroom: float = 1.5,
) -> bool:
    """Decide whether a simulation of ``n_qubits`` should be attempted.

    The check is ``state_size * headroom <= available_bytes``. A headroom of
    1.5 leaves room for simulator workspace (e.g. probability buffers used
    when sampling) so that we skip a size instead of being killed by the OOM
    killer halfway through a run.

    Args:
        n_qubits: Number of qubits.
        precision: ``"fp32"`` or ``"fp64"``.
        available_bytes: Memory we are allowed to use, in bytes.
        headroom: Multiplier applied to the bare statevector size (>= 1).
    """
    if headroom < 1.0:
        raise ValueError(f"headroom must be >= 1.0, got {headroom}")
    if available_bytes <= 0:
        return False
    return estimate_state_memory_bytes(n_qubits, precision) * headroom <= available_bytes


def max_qubits_that_fit(precision: str, available_bytes: int, headroom: float = 1.5) -> int:
    """Largest ``n`` for which :func:`fits_in_memory` is true (0 if none)."""
    n = 0
    while fits_in_memory(n + 1, precision, available_bytes, headroom):
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Host memory (RAM)
# --------------------------------------------------------------------------- #
def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_meminfo() -> dict[str, int]:
    """Parse ``/proc/meminfo`` into a dict of byte counts (Linux only)."""
    text = _read_text(Path("/proc/meminfo"))
    info: dict[str, int] = {}
    if not text:
        return info
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            factor = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
            info[key.strip()] = int(parts[0]) * factor
    return info


def _cgroup_remaining_bytes() -> Optional[int]:
    """Memory left under the tightest cgroup limit of this process.

    Slurm (and Docker) enforce memory limits through cgroups. ``/proc/meminfo``
    shows the whole node, so on a cluster it can report far more memory than
    the job may actually use. We walk the process' cgroup and its parents and
    return the smallest ``limit - usage``; ``None`` means no limit was found.
    """
    text = _read_text(Path("/proc/self/cgroup"))
    if not text:
        return None

    remaining: list[int] = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy_id, controllers, rel_path = parts

        if hierarchy_id == "0" and controllers == "":
            # cgroup v2 (unified hierarchy)
            root = Path("/sys/fs/cgroup")
            limit_file, usage_file = "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            # cgroup v1, memory controller
            root = Path("/sys/fs/cgroup/memory")
            limit_file, usage_file = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue

        directory = root / rel_path.lstrip("/")
        while True:
            limit_txt = _read_text(directory / limit_file)
            if limit_txt and limit_txt != "max" and limit_txt.isdigit():
                limit = int(limit_txt)
                if limit < _CGROUP_V1_UNLIMITED:
                    usage_txt = _read_text(directory / usage_file)
                    usage = int(usage_txt) if usage_txt and usage_txt.isdigit() else 0
                    remaining.append(max(limit - usage, 0))
            if directory == root or root not in directory.parents:
                break
            directory = directory.parent

    return min(remaining) if remaining else None


def _slurm_memory_cap_bytes() -> Optional[int]:
    """Memory requested from Slurm for this job, if it is set and non-zero."""
    for var in ("SLURM_MEM_PER_NODE",):
        value = os.environ.get(var, "")
        if value.isdigit() and int(value) > 0:
            return int(value) * 1024 * 1024  # Slurm reports MiB
    return None


def total_host_memory_bytes() -> int:
    """Total physical RAM of the node in bytes (0 if unknown)."""
    return _read_meminfo().get("MemTotal", 0)


def available_host_memory_bytes() -> int:
    """RAM this process may use right now, in bytes.

    Takes the minimum of the kernel's ``MemAvailable``, the remaining cgroup
    allowance and the Slurm job request, so it is correct both on a laptop
    and inside a memory-limited batch job.
    """
    candidates: list[int] = []
    meminfo = _read_meminfo()
    if "MemAvailable" in meminfo:
        candidates.append(meminfo["MemAvailable"])
    for value in (_cgroup_remaining_bytes(), _slurm_memory_cap_bytes()):
        if value is not None:
            candidates.append(value)
    return min(candidates) if candidates else 0


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
@dataclass
class GpuInfo:
    """One NVIDIA GPU as reported by ``nvidia-smi``."""

    name: str
    total_bytes: int
    free_bytes: int


def query_gpus(timeout_s: float = 20.0) -> list[GpuInfo]:
    """List NVIDIA GPUs via ``nvidia-smi`` (empty list if none or no driver)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    gpus: list[GpuInfo] = []
    for line in result.stdout.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 3:
            continue
        name, total_mib, free_mib = fields
        try:
            gpus.append(
                GpuInfo(name, int(total_mib) * 1024 * 1024, int(free_mib) * 1024 * 1024)
            )
        except ValueError:
            continue
    return gpus


def cudaq_gpu_count() -> int:
    """Number of GPUs CUDA-Q can use (0 if CUDA-Q or a GPU is unavailable)."""
    try:
        import cudaq  # noqa: PLC0415  (lazy import on purpose)

        return int(cudaq.num_available_gpus())
    except Exception:  # noqa: BLE001  any failure means "no usable GPU"
        return 0


def gpu_available() -> bool:
    """True if CUDA-Q reports at least one usable GPU."""
    return cudaq_gpu_count() > 0


def available_gpu_memory_bytes() -> Optional[int]:
    """Free memory on GPU 0 in bytes, or ``None`` if it cannot be measured."""
    gpus = query_gpus()
    return gpus[0].free_bytes if gpus else None


# --------------------------------------------------------------------------- #
# Machine summary (logged with every run, used for the README)
# --------------------------------------------------------------------------- #
def _cpu_model() -> str:
    text = _read_text(Path("/proc/cpuinfo")) or ""
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def physical_core_count() -> int:
    """Physical CPU cores (hyperthreads not counted); falls back to logical."""
    text = _read_text(Path("/proc/cpuinfo")) or ""
    cores: set[tuple[str, str]] = set()
    physical_id = core_id = None
    for line in text.splitlines() + [""]:
        if not line.strip():
            if physical_id is not None and core_id is not None:
                cores.add((physical_id, core_id))
            physical_id = core_id = None
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "physical id":
            physical_id = value.strip()
        elif key == "core id":
            core_id = value.strip()
    return len(cores) or (os.cpu_count() or 1)


def git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Short hash of the checked-out commit, read straight from ``.git``.

    Works on compute nodes where the ``git`` command is not installed.
    Returns ``None`` outside a git checkout.
    """
    root = repo_root or Path(__file__).resolve().parent.parent
    git_dir = root / ".git"
    head = _read_text(git_dir / "HEAD")
    if not head:
        return None
    if not head.startswith("ref:"):
        return head[:7]  # detached HEAD holds the hash itself
    ref = head.split(":", 1)[1].strip()
    sha = _read_text(git_dir / ref)
    if sha is None:  # ref may live in packed-refs after `git gc`
        for line in (_read_text(git_dir / "packed-refs") or "").splitlines():
            if line.endswith(" " + ref):
                sha = line.split()[0]
                break
    return sha[:7] if sha else None


@dataclass
class HardwareInfo:
    """Snapshot of the machine a benchmark ran on."""

    hostname: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    omp_num_threads: Optional[str]
    ram_total_gb: float
    ram_available_gb: float
    gpus: list[dict] = field(default_factory=list)
    cudaq_gpu_count: int = 0
    cudaq_version: Optional[str] = None
    git_commit: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def collect_hardware_info(include_cudaq: bool = True) -> HardwareInfo:
    """Gather a :class:`HardwareInfo` snapshot.

    Args:
        include_cudaq: Also query CUDA-Q (version and GPU count). Set to
            ``False`` for a fast check that does not import CUDA-Q.
    """
    version: Optional[str] = None
    n_gpu = 0
    if include_cudaq:
        try:
            import cudaq  # noqa: PLC0415

            version = str(cudaq.__version__)
            n_gpu = int(cudaq.num_available_gpus())
        except Exception:  # noqa: BLE001
            pass

    return HardwareInfo(
        hostname=platform.node(),
        cpu_model=_cpu_model(),
        physical_cores=physical_core_count(),
        logical_cores=os.cpu_count() or 1,
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        ram_total_gb=round(total_host_memory_bytes() / GIB, 2),
        ram_available_gb=round(available_host_memory_bytes() / GIB, 2),
        gpus=[asdict(g) for g in query_gpus()],
        cudaq_gpu_count=n_gpu,
        cudaq_version=version,
        git_commit=git_commit(),
    )


def main() -> None:
    """Print the hardware summary and the largest qubit counts that fit."""
    info = collect_hardware_info()
    print(json.dumps(info.to_dict(), indent=2))

    ram = available_host_memory_bytes()
    # qpp-cpu peaks at ~2x the statevector (measured, see src/ghz_benchmark.py)
    print(f"\nMax qubits on CPU (fp64, headroom 2.0): {max_qubits_that_fit('fp64', ram, 2.0)}")
    vram = available_gpu_memory_bytes()
    if vram is not None:
        print(f"Max qubits on GPU (fp32, headroom 1.5): {max_qubits_that_fit('fp32', vram)}")
    else:
        print("No NVIDIA GPU detected: GPU benchmarks would be skipped.")


if __name__ == "__main__":
    main()
