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


def test_free_memory_is_measured_once_per_target(monkeypatch, tmp_path):
    """Re-measuring inside the loop would count the simulator's own cache as used."""
    import src.ghz_benchmark as gb

    calls = []
    monkeypatch.setattr(gb, "_available_bytes", lambda spec: calls.append(1) or 64 * 1024**3)
    monkeypatch.setattr(gb, "time_sample", lambda n, shots, warmup, repeats: [0.01] * repeats)
    writer = gb.ResultWriter(tmp_path / "ghz.csv")
    gb.benchmark_target(TARGETS["qpp-cpu"], [10, 11, 12], 10, 0, 2, 2.0, "host", "cpu", None, writer)
    writer.close()
    assert len(calls) == 1
