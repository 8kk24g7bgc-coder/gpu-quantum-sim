"""Tests for the speedup computation used in the GHZ plots."""

from src.plotting import speedup_curves


def _row(device, n, median):
    return {
        "target": "nvidia" if device == "gpu" else "qpp-cpu",
        "n": str(n),
        "median_s": str(median),
        "std_s": "0",
        "precision": "fp32" if device == "gpu" else "fp64",
        "device": device,
        "device_name": "GPU-X" if device == "gpu" else "CPU-Y",
        "status": "ok",
    }


def test_speedup_is_cpu_over_gpu_on_common_sizes():
    rows = [_row("cpu", 20, 2.0), _row("cpu", 21, 4.0), _row("gpu", 21, 0.5), _row("gpu", 22, 0.9)]
    curves = speedup_curves(rows)
    assert list(curves.values()) == [[(21, 8.0)]]


def test_no_gpu_rows_means_no_speedup():
    assert speedup_curves([_row("cpu", 20, 1.0)]) == {}


def test_skipped_rows_are_ignored():
    skipped = _row("gpu", 20, 0.1) | {"status": "skipped_memory", "median_s": ""}
    assert speedup_curves([_row("cpu", 20, 1.0), skipped]) == {}
