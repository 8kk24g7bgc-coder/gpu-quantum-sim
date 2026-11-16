"""Tests for the statevector memory estimator in src.hardware."""

import pytest

from src.hardware import (
    GIB,
    available_host_memory_bytes,
    bytes_per_amplitude,
    estimate_state_memory_bytes,
    estimate_state_memory_gb,
    fits_in_memory,
    max_qubits_that_fit,
)


@pytest.mark.parametrize(
    ("n", "precision", "expected_bytes"),
    [
        (1, "fp32", 16),
        (1, "fp64", 32),
        (20, "fp32", 8 * 2**20),
        (30, "fp32", 8 * GIB),
        (30, "fp64", 16 * GIB),
        (32, "fp64", 64 * GIB),
    ],
)
def test_estimate_matches_2_to_the_n_times_amplitude_size(n, precision, expected_bytes):
    assert estimate_state_memory_bytes(n, precision) == expected_bytes


def test_each_extra_qubit_doubles_memory():
    for n in range(1, 34):
        assert estimate_state_memory_bytes(n + 1, "fp32") == 2 * estimate_state_memory_bytes(n, "fp32")


def test_gb_is_gibibytes():
    assert estimate_state_memory_gb(30, "fp32") == pytest.approx(8.0)
    assert estimate_state_memory_gb(30, "fp64") == pytest.approx(16.0)


def test_fits_in_memory_respects_headroom():
    # 30 qubits fp32 = 8 GiB; with headroom 1.5 it needs 12 GiB.
    assert fits_in_memory(30, "fp32", 12 * GIB, headroom=1.5)
    assert not fits_in_memory(30, "fp32", 12 * GIB - 1, headroom=1.5)
    assert fits_in_memory(30, "fp32", 8 * GIB, headroom=1.0)
    assert not fits_in_memory(5, "fp32", 0)


def test_max_qubits_that_fit():
    # A 16 GB T4 (fp32): 30 qubits need 12 GiB with headroom, 31 need 24 GiB.
    assert max_qubits_that_fit("fp32", 15 * GIB) == 30
    # ~92 GiB node (fp64): 31 qubits need 48 GiB, 32 need 96 GiB.
    assert max_qubits_that_fit("fp64", 92 * GIB) == 31


def test_cpu_headroom_reproduces_measured_oom_limit():
    """Job 51371 on n24 (123 GiB free): n = 31 ran, n = 32 was OOM-killed."""
    from src.ghz_benchmark import TARGETS

    headroom = TARGETS["qpp-cpu"].headroom
    assert fits_in_memory(31, "fp64", 123 * GIB, headroom)
    assert not fits_in_memory(32, "fp64", 123 * GIB, headroom)


@pytest.mark.parametrize(
    "call",
    [
        lambda: estimate_state_memory_bytes(0, "fp32"),
        lambda: bytes_per_amplitude("fp16"),
        lambda: fits_in_memory(5, "fp32", GIB, headroom=0.5),
    ],
)
def test_invalid_input_raises(call):
    with pytest.raises(ValueError):
        call()


def test_available_host_memory_is_positive():
    assert available_host_memory_bytes() > 0
