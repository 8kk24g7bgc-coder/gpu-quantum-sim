"""Render README.md from docs/README.template.md and the files in results/.

Every number in the README is computed here from the result files, so the
README can never drift from the data::

    python scripts/build_readme.py              # writes README.md
    python scripts/build_readme.py --check      # fail if README.md is out of date

Placeholders in the template look like ``{{name}}``; an unknown or unresolved
placeholder is an error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
CHEMICAL_ACCURACY_MHA = 1.6
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def sci(value: float, digits: int = 1) -> str:
    """1.73e-10 -> '1.7 × 10⁻¹⁰' (plain decimal for values in [0.01, 1000))."""
    if value == 0:
        return "0"
    if 0.01 <= abs(value) < 1000:
        return f"{value:.{digits + 1}g}"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa} × 10{str(int(exponent)).translate(_SUPERSCRIPT)}"


def seconds(value: float) -> str:
    """Human-readable duration: '0.885 s', '24.0 min', '1.4 h'."""
    if value < 60:
        return f"{value:.3g} s"
    if value < 3600:
        return f"{value / 60:.1f} min"
    return f"{value / 3600:.1f} h"


def short_cpu_name(model: str) -> str:
    """'Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz' -> 'Intel Xeon E-2388G'."""
    name = re.sub(r"\((R|TM)\)", "", model)
    name = re.sub(r"\bCPU\b|@.*$", "", name)
    return re.sub(r"\s+", " ", name).strip()


def gib(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.0f} GiB"


def energy(value: float, digits: int = 4) -> str:
    """Energy with a typographic minus sign: -1.1373 -> '−1.1373'."""
    return f"{value:.{digits}f}".replace("-", "−")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_ghz(results: Path) -> list[dict]:
    with (results / "ghz_benchmark.csv").open(newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["status"] == "ok" and r["median_s"]]


def load_hardware(results: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results.glob("ghz_benchmark_*_hardware.json"))]


def hardware_for_host(hardware: list[dict], host: str) -> Optional[dict]:
    return next((h for h in hardware if h.get("hostname") == host), None)


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
def ghz_numbers(rows: list[dict], hardware: list[dict]) -> dict[str, str]:
    cpu = {int(r["n"]): r for r in rows if r["device"] == "cpu"}
    gpu = {int(r["n"]): r for r in rows if r["device"] == "gpu"}
    if not cpu or not gpu:
        raise SystemExit("results/ghz_benchmark.csv needs both CPU and GPU rows; run `make plots` first")

    common = sorted(set(cpu) & set(gpu))
    n_cmp = common[-1]
    speedups = {n: float(cpu[n]["median_s"]) / float(gpu[n]["median_s"]) for n in common}
    n_cpu_max, n_gpu_max = max(cpu), max(gpu)

    cpu_row, gpu_row = cpu[n_cpu_max], gpu[n_gpu_max]
    cpu_hw = hardware_for_host(hardware, cpu_row["host"]) or {}
    gpu_hw = hardware_for_host(hardware, gpu_row["host"]) or {}
    gpu_info = (gpu_hw.get("gpus") or [{}])[0]

    return {
        "max_qubits": str(max(n_cpu_max, n_gpu_max)),
        "max_qubits_cpu": str(n_cpu_max),
        "max_qubits_gpu": str(n_gpu_max),
        "max_state_cpu": f"{float(cpu_row['est_mem_gb']):.0f} GiB",
        "max_state_gpu": f"{float(gpu_row['est_mem_gb']):.0f} GiB",
        "next_qubit_cpu": str(n_cpu_max + 1),
        "next_state_cpu": f"{2 * float(cpu_row['est_mem_gb']):.0f} GiB",
        "cpu_time_max": seconds(float(cpu_row["median_s"])),
        "gpu_time_max": seconds(float(gpu_row["median_s"])),
        "n_cmp": str(n_cmp),
        "cpu_time_cmp": seconds(float(cpu[n_cmp]["median_s"])),
        "gpu_time_cmp": seconds(float(gpu[n_cmp]["median_s"])),
        "speedup_cmp": f"{speedups[n_cmp]:,.0f}",
        "speedup_min": f"{min(speedups.values()):,.0f}",
        "n_speedup_min": str(min(speedups, key=speedups.get)),
        "n_min": str(min(min(cpu), min(gpu))),
        "cpu_name": short_cpu_name(cpu_row["device_name"]),
        "cpu_cores": str(cpu_hw.get("physical_cores", "?")),
        "cpu_threads": str(cpu_hw.get("omp_num_threads") or cpu_hw.get("physical_cores", "?")),
        "cpu_ram": f"{cpu_hw.get('ram_total_gb', 0):.0f} GiB",  # hardware.py reports GiB
        "cpu_host": cpu_row["host"],
        "gpu_name": gpu_row["device_name"],
        "gpu_mem": gib(gpu_info.get("total_bytes", 0)),
        "gpu_host_cpu": short_cpu_name(gpu_hw.get("cpu_model", "")),
        "cudaq_version": re.sub(r"^CUDA-Q Version\s+(\S+).*$", r"\1",
                                cpu_hw.get("cudaq_version") or gpu_hw.get("cudaq_version") or "?"),
        "shots": cpu_row["shots"],
        "repeats": cpu_row["repeats"],
    }


def vqe_numbers(summary: dict) -> dict[str, str]:
    max_err = summary["max_abs_error_mHa"]
    if max_err >= CHEMICAL_ACCURACY_MHA:
        raise SystemExit(f"VQE error {max_err} mHa is not within chemical accuracy; refusing to claim it")
    lo, hi, step = summary["r_grid_angstrom"]
    return {
        "max_vqe_error": sci(max_err),
        # how many orders of magnitude below the 1.6 mHa threshold
        "vqe_margin": str(int(math.floor(math.log10(CHEMICAL_ACCURACY_MHA / max_err)))),
        "r_eq": f"{summary['R_eq_angstrom']:.3f}",
        "e_min": energy(summary["E_min_vqe_Ha"]),
        "e_fci_eq": energy(summary["E_fci_at_R_eq_Ha"]),
        "n_pauli": str(summary["n_pauli_terms"]),
        "n_qubits_vqe": str(summary["n_qubits"]),
        "n_params": str(summary["n_parameters"]),
        "n_bonds": str(summary["n_bond_lengths"]),
        "r_min": f"{lo:.2f}",
        "r_max": f"{hi:.2f}",
        "r_step": f"{step:.2f}",
        "e_hf_conv": energy(summary["convergence_E_hf_Ha"]),
        "e_dissociation": energy(summary["dissociation_limit_2E_H_Ha"]),
        "optimizer": {"cobyla": "COBYLA", "lbfgsb": "L-BFGS-B"}.get(summary["optimizer"], summary["optimizer"]),
        "vqe_runtime": seconds(summary["runtime_s"]),
        "vqe_host": summary.get("hardware", {}).get("hostname", "?"),
    }


def collect(results: Path) -> dict[str, str]:
    summary = json.loads((results / "h2_summary.json").read_text())
    return {**ghz_numbers(load_ghz(results), load_hardware(results)), **vqe_numbers(summary)}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render(template: str, values: dict[str, str]) -> str:
    def substitute(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"template placeholder {{{{{key}}}}} has no value")
        return values[key]

    text = re.sub(r"\{\{\s*(\w+)\s*\}\}", substitute, template)
    header = "<!-- Generated by scripts/build_readme.py from results/. Edit docs/README.template.md instead. -->\n"
    return header + text


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--template", type=Path, default=ROOT / "docs" / "README.template.md")
    parser.add_argument("--output", type=Path, default=ROOT / "README.md")
    parser.add_argument("--check", action="store_true", help="exit 1 if --output is out of date")
    args = parser.parse_args(argv)

    text = render(args.template.read_text(), collect(args.results))
    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != text:
            print(f"{args.output} is out of date; run `make readme`", file=sys.stderr)
            return 1
        print(f"{args.output} is up to date")
        return 0
    args.output.write_text(text)
    print(f"README written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
