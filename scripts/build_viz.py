"""Build the interactive demo page (docs/index.html) from the project's results.

The page has three panels: the memory wall (measured CPU/GPU timings), a live
GHZ statevector simulation, and an H2 VQE explorer. Everything the page shows
as *measured* is read from results/; the H2 Hamiltonians are rebuilt with the
same code as the VQE run (src/h2_vqe.py) and checked against FCI before they
are written, so the in-browser energies are exact::

    python scripts/build_viz.py          # writes docs/index.html

Requires the project environment (cudaq, pyscf) for the Hamiltonians.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ghz_benchmark import TARGETS  # noqa: E402
from src.h2_vqe import N_ELECTRONS, N_QUBITS, build_hamiltonian  # noqa: E402
from src.hardware import git_commit  # noqa: E402

MARKER = "/*__VIZ_DATA__*/null"
_PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}


# --------------------------------------------------------------------------- #
# H2 Hamiltonian in a form the browser can use
# --------------------------------------------------------------------------- #
def pauli_terms(r: float) -> list[tuple[str, float]]:
    """Pauli words and real coefficients of the H2 qubit Hamiltonian at R.

    Character k of a word acts on qubit k (CUDA-Q convention).
    """
    terms = []
    for term in build_hamiltonian(r):
        coef = complex(term.evaluate_coefficient())
        if abs(coef.imag) > 1e-12:
            raise ValueError(f"unexpected complex coefficient {coef} at R={r}")
        terms.append((term.get_pauli_word(N_QUBITS), float(coef.real)))
    return terms


def matrix_from_terms(terms: list[tuple[str, float]], n_qubits: int = N_QUBITS) -> np.ndarray:
    """Dense matrix with qubit 0 as the least significant bit of the index.

    This is the same convention the JavaScript code uses, and it reproduces
    ``SpinOperator.to_matrix()`` exactly (checked in tests/test_viz.py).
    """
    dim = 2**n_qubits
    out = np.zeros((dim, dim), dtype=complex)
    for word, coef in terms:
        op = np.array([[1.0 + 0j]])
        for ch in reversed(word):  # kron(q_{n-1}, ..., q_0)
            op = np.kron(op, _PAULI[ch])
        out += coef * op
    return out


def reference_indices() -> tuple[int, int]:
    """Basis indices of the Hartree-Fock state |0011> and the double excitation |1100>."""
    hf = sum(1 << k for k in range(N_ELECTRONS))
    double = sum(1 << k for k in range(N_QUBITS - N_ELECTRONS, N_QUBITS))
    return hf, double


def two_level_energies(terms: list[tuple[str, float]]) -> tuple[float, float, float]:
    """(H_aa, H_bb, H_ab) in the {HF, doubly excited} subspace.

    For H2 the ground state lives in this two-dimensional subspace, so
    E(theta) = <psi(theta)|H|psi(theta)> with
    |psi(theta)> = cos(theta)|HF> + sin(theta)|double> reaches FCI exactly.
    """
    m = matrix_from_terms(terms)
    a, b = reference_indices()
    return float(m[a, a].real), float(m[b, b].real), float(m[a, b].real)


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def ghz_data(results: Path, docs: Path) -> dict:
    rows = [r for r in _read_csv(results / "ghz_benchmark.csv") if r["status"] == "ok" and r["median_s"]]
    hardware = [json.loads(p.read_text()) for p in sorted(results.glob("ghz_benchmark_*_hardware.json"))]

    def series(device: str) -> dict:
        sel = sorted((r for r in rows if r["device"] == device), key=lambda r: int(r["n"]))
        if not sel:
            return {}
        host = sel[0]["host"]
        hw = next((h for h in hardware if h.get("hostname") == host), {})
        return {
            "name": sel[0]["device_name"],
            "host": host,
            "precision": sel[0]["precision"],
            "points": [[int(r["n"]), float(r["median_s"]), float(r["std_s"] or 0)] for r in sel],
            "hardware": hw,
        }

    cpu, gpu = series("cpu"), series("gpu")
    oom_n: Optional[int] = None
    log = docs / "ghz_cpu_n24_job51371.log"
    if cpu and log.exists() and "Killed" in log.read_text(errors="ignore"):
        oom_n = cpu["points"][-1][0] + 1  # the run was killed while attempting the next size

    gpu_info = (gpu.get("hardware", {}).get("gpus") or [{}])[0] if gpu else {}
    return {
        "cpu": cpu,
        "gpu": gpu,
        "cpu_ram_gib": cpu.get("hardware", {}).get("ram_total_gb") if cpu else None,
        "gpu_mem_gib": gpu_info.get("total_bytes", 0) / 1024**3 if gpu_info else None,
        "cpu_oom_n": oom_n,
        "headroom": {"cpu": TARGETS["qpp-cpu"].headroom, "gpu": TARGETS["nvidia"].headroom},
        "shots": int(rows[0]["shots"]),
        "repeats": int(rows[0]["repeats"]),
    }


def h2_data(results: Path, tol: float = 1e-8) -> dict:
    pes = _read_csv(results / "h2_pes.csv")
    summary = json.loads((results / "h2_summary.json").read_text())
    radii = [float(r["R"]) for r in pes]
    terms_per_r, blocks = [], []
    for r, row in zip(radii, pes):
        terms = pauli_terms(r)
        haa, hbb, hab = two_level_energies(terms)
        e_min = 0.5 * (haa + hbb) - np.hypot(0.5 * (haa - hbb), hab)
        e_fci = float(row["E_fci"])
        if abs(e_min - e_fci) > tol or abs(haa - float(row["E_hf"])) > tol:
            raise SystemExit(f"two-level model disagrees with PySCF at R={r}: "
                             f"min {e_min} vs FCI {e_fci}, HF {haa} vs {row['E_hf']}")
        terms_per_r.append([[w, round(c, 12)] for w, c in terms])
        blocks.append([haa, hbb, hab])
    hf, double = reference_indices()
    return {
        "R": radii,
        "E_vqe": [float(r["E_vqe"]) for r in pes],
        "E_fci": [float(r["E_fci"]) for r in pes],
        "E_hf": [float(r["E_hf"]) for r in pes],
        "terms": terms_per_r,
        "block": blocks,
        "hf_index": hf,
        "double_index": double,
        "n_qubits": N_QUBITS,
        "R_eq": summary["R_eq_angstrom"],
        "E_min": summary["E_min_vqe_Ha"],
        "E_dissociation": summary["dissociation_limit_2E_H_Ha"],
        "max_error_mHa": summary["max_abs_error_mHa"],
        "optimizer": summary["optimizer"],
    }


def collect(results: Path, docs: Path) -> dict:
    return {
        "commit": git_commit(),
        "repo": "https://github.com/8kk24g7bgc-coder/gpu-quantum-sim",
        "ghz": ghz_data(results, docs),
        "h2": h2_data(results),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--template", type=Path, default=ROOT / "docs" / "viz.template.html")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "index.html")
    args = parser.parse_args(argv)

    template = args.template.read_text()
    if MARKER not in template:
        raise SystemExit(f"{args.template} has no {MARKER} marker")
    data = collect(args.results, ROOT / "docs")
    args.output.write_text(template.replace(MARKER, json.dumps(data, separators=(",", ":"))))
    print(f"Demo page written to {args.output} "
          f"({len(data['h2']['R'])} bond lengths, {len(data['ghz']['cpu']['points'])} CPU "
          f"and {len(data['ghz']['gpu']['points'])} GPU points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
