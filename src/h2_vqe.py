"""VQE with a UCCSD ansatz for H2 (STO-3G), validated against exact FCI.

Pipeline for one bond length R:

1. Molecule -> qubit Hamiltonian. ``cudaq.chemistry.create_molecular_hamiltonian``
   runs PySCF + OpenFermion and applies the Jordan-Wigner mapping: 4 spin
   orbitals -> 4 qubits, 15 Pauli terms (including the identity, which holds
   the nuclear repulsion and core energy).
2. Ansatz. Start from the Hartree-Fock state |1100> and apply the UCCSD
   circuit for 2 electrons in 4 spin orbitals (3 parameters).
3. Classical loop. SciPy minimises the energy <psi(theta)|H|psi(theta)>,
   evaluated with ``cudaq.observe``; every evaluation is recorded.

The script sweeps R from 0.30 to 2.50 Angstrom (warm-starting each point from
the previous optimum), compares every point with FCI and Hartree-Fock from
PySCF, refines the equilibrium bond length, and writes CSV/JSON results and
plots.

Usage (from the repository root)::

    python -m src.h2_vqe            # full run (45 bond lengths + refinement)
    python -m src.h2_vqe --quick    # reduced run, well under 2 minutes on a CPU
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

warnings.filterwarnings("ignore", message=r"cudaq-logical is in preview", category=UserWarning)
warnings.filterwarnings("ignore", message=r"The CUDA-Q `sample` and `observe`", category=FutureWarning)

import cudaq  # noqa: E402
import numpy as np  # noqa: E402
from cudaq.kernels import uccsd  # noqa: E402
from pyscf import fci, gto, scf  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402

from src.hardware import collect_hardware_info  # noqa: E402

N_QUBITS: int = 4
N_ELECTRONS: int = 2
BASIS: str = "sto-3g"
CHEMICAL_ACCURACY_HA: float = 1.6e-3  # 1 kcal/mol ~ 1.6 mHa
R_EQUILIBRIUM_REFERENCE: float = 0.735  # Angstrom, used for the convergence plot

PES_FIELDS: list[str] = [
    "R",
    "E_vqe",
    "E_fci",
    "E_hf",
    "abs_error_mHa",
    "n_evals",
    "converged",
]


# --------------------------------------------------------------------------- #
# Hamiltonian and reference energies
# --------------------------------------------------------------------------- #
def h2_geometry(r_angstrom: float) -> list[tuple[str, tuple[float, float, float]]]:
    """H2 along the z axis with bond length ``r_angstrom``."""
    return [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, float(r_angstrom)))]


@contextlib.contextmanager
def _in_temporary_directory() -> Iterator[None]:
    """Run a block inside a throw-away directory.

    OpenFermion-PySCF saves an ``.hdf5`` file for every molecule in the current
    directory; a 45-point sweep would otherwise litter the repository.
    """
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="h2_vqe_") as tmp:
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(previous)


def build_hamiltonian(r_angstrom: float) -> cudaq.SpinOperator:
    """Jordan-Wigner qubit Hamiltonian of H2/STO-3G at bond length R (4 qubits)."""
    with _in_temporary_directory():
        hamiltonian, _ = cudaq.chemistry.create_molecular_hamiltonian(
            h2_geometry(r_angstrom), BASIS, 1, 0
        )
    return hamiltonian


def pauli_term_count(hamiltonian: cudaq.SpinOperator) -> int:
    """Number of Pauli strings in the Hamiltonian, identity included."""
    return int(hamiltonian.term_count)


def exact_ground_energy(hamiltonian: cudaq.SpinOperator) -> float:
    """Lowest eigenvalue of the 16x16 qubit Hamiltonian (exact diagonalisation)."""
    matrix = np.array(hamiltonian.to_matrix())
    return float(np.linalg.eigvalsh(matrix).min())


@dataclass(frozen=True)
class ReferenceEnergies:
    """Classical reference energies from PySCF, in Hartree."""

    hf: float
    fci: float


def reference_energies(r_angstrom: float) -> ReferenceEnergies:
    """Restricted Hartree-Fock and exact FCI energies of H2/STO-3G from PySCF."""
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {r_angstrom}", basis=BASIS, unit="Angstrom", verbose=0
    )
    mean_field = scf.RHF(mol).run()
    e_fci, _ = fci.FCI(mean_field).kernel()
    return ReferenceEnergies(hf=float(mean_field.e_tot), fci=float(e_fci))


def hydrogen_atom_energy() -> float:
    """Energy of one H atom in STO-3G (UHF is exact for one electron).

    ``2 * hydrogen_atom_energy()`` is the dissociation limit of H2 in this
    basis, drawn as the asymptote of the potential energy curve.
    """
    atom = gto.M(atom="H 0 0 0", basis=BASIS, spin=1, verbose=0)
    return float(scf.UHF(atom).run().e_tot)


# --------------------------------------------------------------------------- #
# Ansatz and VQE
# --------------------------------------------------------------------------- #
@cudaq.kernel
def uccsd_ansatz(thetas: list[float], n_qubits: int, n_electrons: int):
    """Hartree-Fock reference |1..10..0> followed by the UCCSD circuit."""
    q = cudaq.qvector(n_qubits)
    for i in range(n_electrons):
        x(q[i])
    uccsd(q, thetas, n_electrons, n_qubits)


def num_parameters() -> int:
    """UCCSD parameters for 2 electrons in 4 spin orbitals (3 in CUDA-Q 0.16)."""
    return int(cudaq.kernels.uccsd_num_parameters(N_ELECTRONS, N_QUBITS))


def energy(hamiltonian: cudaq.SpinOperator, thetas: Sequence[float]) -> float:
    """Expectation value <psi(thetas)|H|psi(thetas)> from ``cudaq.observe``."""
    result = cudaq.observe(uccsd_ansatz, hamiltonian, [float(t) for t in thetas], N_QUBITS, N_ELECTRONS)
    return float(result.expectation())


@dataclass
class VQEResult:
    """Outcome of one VQE optimisation."""

    energy: float
    parameters: list[float]
    history: list[float] = field(default_factory=list)  # energy at every evaluation
    n_evals: int = 0
    converged: bool = False


OPTIMIZERS: dict[str, dict] = {
    # Derivative-free; rhobeg is the initial trust-region radius in radians.
    "cobyla": {"method": "COBYLA", "tol": 1e-10, "options": {"maxiter": 2000, "rhobeg": 0.2}},
    # Quasi-Newton with finite-difference gradients (exact statevector => smooth).
    "lbfgsb": {"method": "L-BFGS-B", "options": {"ftol": 1e-14, "gtol": 1e-9, "maxiter": 2000}},
}


def run_vqe(
    hamiltonian: cudaq.SpinOperator,
    initial_parameters: Optional[Sequence[float]] = None,
    optimizer: str = "cobyla",
) -> VQEResult:
    """Minimise the UCCSD energy with SciPy and record every evaluation.

    Args:
        hamiltonian: Qubit Hamiltonian.
        initial_parameters: Starting angles; zeros (= Hartree-Fock) if ``None``.
        optimizer: Key of :data:`OPTIMIZERS` (``"cobyla"`` or ``"lbfgsb"``).
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {sorted(OPTIMIZERS)}, got {optimizer!r}")
    x0 = np.zeros(num_parameters()) if initial_parameters is None else np.asarray(initial_parameters, float)

    history: list[float] = []

    def objective(thetas: np.ndarray) -> float:
        value = energy(hamiltonian, thetas)
        history.append(value)
        return value

    result = minimize(objective, x0, **OPTIMIZERS[optimizer])
    return VQEResult(
        energy=float(result.fun),
        parameters=[float(t) for t in result.x],
        history=history,
        n_evals=len(history),
        converged=bool(result.success),
    )


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def bond_lengths(start: float, stop: float, step: float) -> list[float]:
    """Inclusive grid of bond lengths, rounded to avoid float drift (0.30, 0.35, ...)."""
    n = int(round((stop - start) / step))
    return [round(start + i * step, 4) for i in range(n + 1)]


def sweep_pes(
    radii: Sequence[float],
    optimizer: str,
    on_point: Optional[Callable[[dict], None]] = None,
) -> tuple[list[dict], list[float]]:
    """VQE over a list of bond lengths, warm-starting from the previous optimum.

    Returns:
        The rows (one dict per R) and the optimal parameters at the last R.
    """
    rows: list[dict] = []
    params: Optional[list[float]] = None
    for r in radii:
        hamiltonian = build_hamiltonian(r)
        vqe = run_vqe(hamiltonian, params, optimizer)
        params = vqe.parameters
        ref = reference_energies(r)
        row = {
            "R": r,
            "E_vqe": vqe.energy,
            "E_fci": ref.fci,
            "E_hf": ref.hf,
            "abs_error_mHa": abs(vqe.energy - ref.fci) * 1e3,
            "n_evals": vqe.n_evals,
            "converged": vqe.converged,
        }
        rows.append(row)
        if on_point is not None:
            on_point(row)
    return rows, params or []


def refine_equilibrium(
    rows: list[dict], optimizer: str, xatol: float = 1e-4
) -> dict:
    """Locate the VQE minimum of the energy curve between the grid points.

    Takes the lowest grid point, then minimises E_vqe(R) on the interval to its
    neighbours with a bounded scalar search; every evaluation is a full VQE.
    """
    i_min = int(np.argmin([row["E_vqe"] for row in rows]))
    lo = rows[max(i_min - 1, 0)]["R"]
    hi = rows[min(i_min + 1, len(rows) - 1)]["R"]
    cache: dict[float, VQEResult] = {}

    def vqe_energy(r: float) -> float:
        cache[r] = run_vqe(build_hamiltonian(r), None, optimizer)
        return cache[r].energy

    result = minimize_scalar(vqe_energy, bounds=(lo, hi), method="bounded", options={"xatol": xatol})
    r_eq = float(result.x)
    e_vqe = cache[r_eq].energy if r_eq in cache else vqe_energy(r_eq)
    ref = reference_energies(r_eq)
    return {
        "R_eq_angstrom": r_eq,
        "E_min_vqe_Ha": e_vqe,
        "E_fci_at_R_eq_Ha": ref.fci,
        "abs_error_at_R_eq_mHa": abs(e_vqe - ref.fci) * 1e3,
        "search_interval_angstrom": [lo, hi],
    }


def convergence_run(r: float, optimizer: str) -> tuple[VQEResult, ReferenceEnergies, int]:
    """VQE from the Hartree-Fock point (zero angles) at one bond length."""
    hamiltonian = build_hamiltonian(r)
    return run_vqe(hamiltonian, None, optimizer), reference_energies(r), pauli_term_count(hamiltonian)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_pes_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PES_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "E_vqe": f"{row['E_vqe']:.10f}",
                "E_fci": f"{row['E_fci']:.10f}",
                "E_hf": f"{row['E_hf']:.10f}",
                "abs_error_mHa": f"{row['abs_error_mHa']:.6e}",
            })


def write_convergence_csv(result: VQEResult, e_fci: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["evaluation", "energy_Ha", "abs_error_mHa"])
        for i, value in enumerate(result.history, start=1):
            writer.writerow([i, f"{value:.10f}", f"{abs(value - e_fci) * 1e3:.6e}"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--r-min", type=float, default=0.30, help="first bond length [Angstrom]")
    parser.add_argument("--r-max", type=float, default=2.50, help="last bond length [Angstrom]")
    parser.add_argument("--r-step", type=float, default=0.05, help="bond-length step [Angstrom]")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="cobyla")
    parser.add_argument("--target", default="qpp-cpu", help="CUDA-Q target (default: qpp-cpu)")
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true", help="do not write the PNG plots")
    parser.add_argument("--quick", action="store_true", help="coarse grid (step 0.20 Angstrom)")
    args = parser.parse_args(argv)
    if args.quick:
        args.r_step = 0.20
    if not (0 < args.r_min < args.r_max) or args.r_step <= 0:
        parser.error("need 0 < --r-min < --r-max and --r-step > 0")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    hw = collect_hardware_info()
    cudaq.set_target(args.target)
    out = args.outdir
    print(f"H2 / {BASIS} / Jordan-Wigner / UCCSD-VQE  (target={args.target}, optimizer={args.optimizer})")

    # 1. Convergence at the reference equilibrium bond length.
    conv, ref, n_terms = convergence_run(R_EQUILIBRIUM_REFERENCE, args.optimizer)
    print(f"Hamiltonian at R={R_EQUILIBRIUM_REFERENCE} A: {n_terms} Pauli terms (identity included), "
          f"{N_QUBITS} qubits, {num_parameters()} UCCSD parameters")
    print(f"Convergence at R={R_EQUILIBRIUM_REFERENCE} A: E_vqe={conv.energy:.8f} Ha  "
          f"E_fci={ref.fci:.8f} Ha  E_hf={ref.hf:.8f} Ha  "
          f"|error|={abs(conv.energy - ref.fci) * 1e3:.2e} mHa  ({conv.n_evals} evaluations)")
    write_convergence_csv(conv, ref.fci, out / "vqe_convergence.csv")

    # 2. Potential energy surface with warm starts.
    radii = bond_lengths(args.r_min, args.r_max, args.r_step)
    print(f"\nSweeping {len(radii)} bond lengths from {radii[0]} to {radii[-1]} A ...")

    def report(row: dict) -> None:
        print(f"  R={row['R']:.2f} A  E_vqe={row['E_vqe']:.8f}  E_fci={row['E_fci']:.8f}  "
              f"E_hf={row['E_hf']:.8f}  |err|={row['abs_error_mHa']:.2e} mHa  ({row['n_evals']} evals)")

    rows, _ = sweep_pes(radii, args.optimizer, on_point=report)
    write_pes_csv(rows, out / "h2_pes.csv")

    # 3. Equilibrium geometry from VQE.
    eq = refine_equilibrium(rows, args.optimizer)
    max_error = max(row["abs_error_mHa"] for row in rows)
    e_h = hydrogen_atom_energy()

    summary = {
        "molecule": "H2",
        "basis": BASIS,
        "mapping": "Jordan-Wigner",
        "ansatz": "UCCSD",
        "optimizer": args.optimizer,
        "target": args.target,
        "n_qubits": N_QUBITS,
        "n_pauli_terms": n_terms,
        "n_parameters": num_parameters(),
        "n_bond_lengths": len(rows),
        "r_grid_angstrom": [radii[0], radii[-1], args.r_step],
        "convergence_R_angstrom": R_EQUILIBRIUM_REFERENCE,
        "convergence_E_vqe_Ha": conv.energy,
        "convergence_E_fci_Ha": ref.fci,
        "convergence_E_hf_Ha": ref.hf,
        "convergence_n_evals": conv.n_evals,
        "max_abs_error_mHa": max_error,
        "all_within_chemical_accuracy": bool(max_error < CHEMICAL_ACCURACY_HA * 1e3),
        "chemical_accuracy_mHa": CHEMICAL_ACCURACY_HA * 1e3,
        "E_H_atom_Ha": e_h,
        "dissociation_limit_2E_H_Ha": 2 * e_h,
        **eq,
        "runtime_s": round(time.perf_counter() - started, 1),
        "hardware": hw.to_dict(),
    }
    (out / "h2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nEquilibrium (VQE): R_eq = {eq['R_eq_angstrom']:.4f} A, "
          f"E_min = {eq['E_min_vqe_Ha']:.6f} Ha  (FCI there: {eq['E_fci_at_R_eq_Ha']:.6f} Ha)")
    print(f"Max |E_vqe - E_fci| over the curve: {max_error:.2e} mHa "
          f"({'within' if summary['all_within_chemical_accuracy'] else 'NOT within'} chemical accuracy)")
    print(f"Results written to {out}/h2_pes.csv, {out}/vqe_convergence.csv, {out}/h2_summary.json")

    if not args.no_plot:
        from src.plotting import plot_vqe_results  # noqa: PLC0415

        for path in plot_vqe_results(out):
            print(f"Plot written to {path}")
    print(f"Total runtime: {summary['runtime_s']} s")
    return 0 if summary["all_within_chemical_accuracy"] else 1


if __name__ == "__main__":
    sys.exit(main())
