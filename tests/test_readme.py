"""Tests for the README builder: numbers must come from the data, never be invented."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_readme as br  # noqa: E402


def test_formatting_helpers():
    assert br.sci(1.73e-10) == "1.7 × 10⁻¹⁰"
    assert br.sci(0.5) == "0.5"
    assert br.seconds(0.885045) == "0.885 s"
    assert br.seconds(1440.92) == "24.0 min"
    assert br.short_cpu_name("Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz") == "Intel Xeon E-2388G"
    assert br.energy(-1.13730604) == "−1.1373"


def test_unresolved_placeholder_is_an_error():
    with pytest.raises(KeyError):
        br.render("{{missing}}", {})


def test_vqe_numbers_refuse_to_claim_chemical_accuracy_when_it_is_not_met():
    summary = {"max_abs_error_mHa": 2.0, "r_grid_angstrom": [0.3, 2.5, 0.05]}
    with pytest.raises(SystemExit):
        br.vqe_numbers(summary)


def _write_ghz(path: Path, rows: list[tuple]) -> None:
    fields = ["target", "n", "median_s", "std_s", "est_mem_gb", "precision", "device",
              "device_name", "host", "shots", "repeats", "status"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        w.writerows(rows)


def test_ghz_headline_numbers_are_computed_from_the_csv(tmp_path):
    rows = [
        ("qpp-cpu", 20, 1.0, 0, 0.0156, "fp64", "cpu", "CPU-X", "h1", 1000, 5, "ok"),
        ("qpp-cpu", 21, 2.0, 0, 0.0312, "fp64", "cpu", "CPU-X", "h1", 1000, 5, "ok"),
        ("qpp-cpu", 22, 4.0, 0, 0.0625, "fp64", "cpu", "CPU-X", "h1", 1000, 5, "ok"),
        ("nvidia", 20, 0.01, 0, 0.0078, "fp32", "gpu", "GPU-Y", "h2", 1000, 5, "ok"),
        ("nvidia", 21, 0.01, 0, 0.0156, "fp32", "gpu", "GPU-Y", "h2", 1000, 5, "ok"),
        ("nvidia", 22, 0.0, 0, 0.0312, "fp32", "gpu", "GPU-Y", "h2", 1000, 5, "skipped_memory"),
    ]
    _write_ghz(tmp_path / "ghz_benchmark.csv", rows)
    hw = [{"hostname": "h1", "physical_cores": 8, "ram_total_gb": 125.0, "cudaq_version": "CUDA-Q Version 0.16.0 (x)"},
          {"hostname": "h2", "gpus": [{"name": "GPU-Y", "total_bytes": 16 * 1024**3}], "cpu_model": "Z"}]
    numbers = br.ghz_numbers(br.load_ghz(tmp_path), hw)
    assert numbers["max_qubits"] == "22"          # largest n with status ok (CPU)
    assert numbers["max_qubits_gpu"] == "21"      # skipped row is ignored
    assert numbers["n_cmp"] == "21"               # largest n measured on both devices
    assert numbers["speedup_cmp"] == "200"        # 2.0 s / 0.01 s
    assert numbers["cudaq_version"] == "0.16.0"
    assert numbers["cpu_ram"] == "125 GiB"
