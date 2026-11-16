"""Tests for the H2 UCCSD-VQE experiment."""

import os

import pytest

cudaq = pytest.importorskip("cudaq")
pytest.importorskip("pyscf")

from src.h2_vqe import (  # noqa: E402
    CHEMICAL_ACCURACY_HA,
    bond_lengths,
    build_hamiltonian,
    energy,
    exact_ground_energy,
    hydrogen_atom_energy,
    num_parameters,
    pauli_term_count,
    reference_energies,
    run_vqe,
)

R_EQ = 0.735


@pytest.fixture(autouse=True, scope="module")
def cpu_target():
    cudaq.set_target("qpp-cpu")


@pytest.fixture(scope="module")
def h_eq():
    return build_hamiltonian(R_EQ)


def test_jordan_wigner_hamiltonian_has_15_pauli_terms(h_eq):
    assert pauli_term_count(h_eq) == 15


@pytest.mark.parametrize("r", [R_EQ, 1.5, 2.5])
def test_qubit_hamiltonian_ground_state_equals_pyscf_fci(r):
    """Exact diagonalisation of the 4-qubit Hamiltonian reproduces FCI."""
    assert exact_ground_energy(build_hamiltonian(r)) == pytest.approx(reference_energies(r).fci, abs=1e-8)


def test_zero_angles_give_the_hartree_fock_energy(h_eq):
    assert energy(h_eq, [0.0] * num_parameters()) == pytest.approx(reference_energies(R_EQ).hf, abs=1e-8)


@pytest.mark.parametrize("optimizer", ["cobyla", "lbfgsb"])
def test_vqe_at_equilibrium_is_within_chemical_accuracy_of_fci(h_eq, optimizer):
    result = run_vqe(h_eq, optimizer=optimizer)
    e_fci = reference_energies(R_EQ).fci
    assert abs(result.energy - e_fci) < CHEMICAL_ACCURACY_HA
    assert result.energy == pytest.approx(-1.1373, abs=1e-4)  # literature value, STO-3G
    assert result.n_evals == len(result.history) > 0


def test_vqe_energy_is_variational(h_eq):
    """No evaluation may go below the exact ground state (variational principle)."""
    result = run_vqe(h_eq)
    assert min(result.history) >= exact_ground_energy(h_eq) - 1e-9


def test_bond_length_grid_matches_the_spec():
    radii = bond_lengths(0.30, 2.50, 0.05)
    assert len(radii) == 45 and radii[0] == 0.30 and radii[-1] == 2.50 and 0.75 in radii


def test_dissociation_limit_is_two_hydrogen_atoms():
    e_h = hydrogen_atom_energy()
    assert e_h == pytest.approx(-0.46658, abs=1e-4)  # H atom in STO-3G
    assert reference_energies(2.5).fci == pytest.approx(2 * e_h, abs=5e-3)


def test_hamiltonian_builder_leaves_no_files_behind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_hamiltonian(0.9)
    assert os.listdir(tmp_path) == []
