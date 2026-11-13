"""Correctness tests for the GHZ kernel on the CPU target."""

import pytest

cudaq = pytest.importorskip("cudaq")

from src.ghz_benchmark import TARGETS, check_ghz_correctness, ghz_counts_are_valid, sample_ghz  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_target():
    TARGETS["qpp-cpu"].activate()


@pytest.mark.parametrize("n", [2, 5, 8])
def test_ghz_only_produces_all_zeros_and_all_ones(n):
    counts = sample_ghz(n, shots=4000)
    assert set(counts) <= {"0" * n, "1" * n}
    assert sum(counts.values()) == 4000


def test_ghz_outcomes_are_balanced():
    counts = check_ghz_correctness(n=8, shots=4000, tolerance=0.05)
    assert ghz_counts_are_valid(counts, 8)


def test_validator_rejects_wrong_bitstrings():
    assert not ghz_counts_are_valid({"000": 500, "111": 400, "010": 100}, 3)


def test_validator_rejects_unbalanced_counts():
    assert not ghz_counts_are_valid({"000": 900, "111": 100}, 3)
    assert ghz_counts_are_valid({"000": 510, "111": 490}, 3)
