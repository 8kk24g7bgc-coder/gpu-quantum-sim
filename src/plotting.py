"""Publication-quality plots for the benchmark and VQE results.

Plots are built only from the CSV/JSON files the experiments write, so they
can be regenerated at any time without re-running a simulation::

    python -m src.plotting ghz --merge-to results/ghz_benchmark.csv   # all ghz_benchmark_*.csv
    python -m src.plotting vqe --outdir results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # headless: works on cluster nodes and in CI
import matplotlib.pyplot as plt  # noqa: E402

NVIDIA_GREEN = "#76B900"
CPU_COLORS = ["#4C72B0", "#8172B2", "#64B5CD", "#937860"]
GPU_COLORS = [NVIDIA_GREEN, "#2E7D32", "#9CCC65"]

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
}

Series = dict[str, list[tuple[int, float, float]]]  # label -> [(n, median, std)]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_ghz_rows(paths: Iterable[Path]) -> list[dict]:
    """Read and concatenate GHZ benchmark CSVs."""
    rows: list[dict] = []
    for path in paths:
        with Path(path).open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def merge_ghz_csvs(paths: Iterable[Path], out_path: Path) -> Path:
    """Concatenate per-machine GHZ CSVs into one file (``results/ghz_benchmark.csv``).

    Each machine writes its own ``ghz_benchmark_<tag>.csv``; the merged file is
    the single raw-results table the README and plots are built from.
    """
    out_path = Path(out_path)
    sources = [Path(p) for p in paths if Path(p).resolve() != out_path.resolve()]
    rows = load_ghz_rows(sources)
    if not rows:
        raise ValueError("no GHZ rows found in " + ", ".join(map(str, sources)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _series_label(row: dict) -> str:
    kind = "GPU" if row["device"] == "gpu" else "CPU"
    return f"{kind} · {row['target']} ({row['precision']}) · {row['device_name']}"


def ghz_series(rows: list[dict]) -> tuple[Series, Series]:
    """Split successful rows into CPU and GPU series keyed by a readable label."""
    cpu: Series = defaultdict(list)
    gpu: Series = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok" or not row.get("median_s"):
            continue
        point = (int(row["n"]), float(row["median_s"]), float(row["std_s"] or 0.0))
        (gpu if row["device"] == "gpu" else cpu)[_series_label(row)].append(point)
    for series in (cpu, gpu):
        for points in series.values():
            points.sort()
    return dict(cpu), dict(gpu)


# --------------------------------------------------------------------------- #
# GHZ plots
# --------------------------------------------------------------------------- #
def plot_ghz_runtime(rows: list[dict], out_path: Path) -> Path:
    """Runtime vs number of qubits (log y), one line per target/device."""
    cpu, gpu = ghz_series(rows)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for colors, series, marker in ((CPU_COLORS, cpu, "o"), (GPU_COLORS, gpu, "s")):
            for color, (label, points) in zip(colors, series.items()):
                n, med, std = zip(*points)
                ax.errorbar(n, med, yerr=std, marker=marker, ms=5, lw=1.8, capsize=3,
                            color=color, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Number of qubits $n$")
        ax.set_ylabel("Wall-clock time per run [s] (median)")
        ax.set_title("GHZ statevector simulation: CPU vs GPU")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=9, loc="upper left")
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def speedup_curves(rows: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """CPU-median / GPU-median at every n measured on both, per (CPU, GPU) pair."""
    cpu, gpu = ghz_series(rows)
    curves: dict[str, list[tuple[int, float]]] = {}
    for cpu_label, cpu_points in cpu.items():
        cpu_by_n = {n: med for n, med, _ in cpu_points}
        for gpu_label, gpu_points in gpu.items():
            pairs = [(n, cpu_by_n[n] / med) for n, med, _ in gpu_points if n in cpu_by_n and med > 0]
            if pairs:
                cpu_name = cpu_label.split(" · ")[-1]
                gpu_name = gpu_label.split(" · ")[-1]
                curves[f"{gpu_name} vs {cpu_name}"] = sorted(pairs)
    return curves


def plot_ghz_speedup(rows: list[dict], out_path: Path) -> Optional[Path]:
    """GPU speedup (CPU time / GPU time) vs qubits; ``None`` if no GPU data."""
    curves = speedup_curves(rows)
    if not curves:
        return None
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for color, (label, points) in zip(GPU_COLORS + CPU_COLORS, curves.items()):
            n, s = zip(*points)
            ax.plot(n, s, marker="o", lw=2, color=color, label=label)
            ax.annotate(f"{s[-1]:.0f}×", (n[-1], s[-1]), textcoords="offset points",
                        xytext=(6, 4), fontsize=10, color=color, fontweight="bold")
        ax.axhline(1.0, color="0.5", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("Number of qubits $n$")
        ax.set_ylabel("Speedup (CPU time / GPU time)")
        ax.set_title("GPU speedup on GHZ circuits")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=9, loc="upper left")
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_ghz_results(csv_paths: Iterable[Path], out_dir: Path) -> list[Path]:
    """Write ``ghz_benchmark.png`` and, if GPU data exist, ``ghz_speedup.png``."""
    rows = load_ghz_rows(csv_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [plot_ghz_runtime(rows, out_dir / "ghz_benchmark.png")]
    speedup = plot_ghz_speedup(rows, out_dir / "ghz_speedup.png")
    if speedup is None:
        print("No GPU results found: ghz_speedup.png not written.")
    else:
        written.append(speedup)
    return written


# --------------------------------------------------------------------------- #
# VQE plots
# --------------------------------------------------------------------------- #
CHEMICAL_ACCURACY_MHA = 1.6
FCI_COLOR = "#222222"
HF_COLOR = "#C44E52"


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def plot_vqe_convergence(csv_path: Path, summary: dict, out_path: Path) -> Path:
    """Energy and error vs optimizer evaluation at the reference bond length."""
    rows = _read_csv(csv_path)
    it = [int(r["evaluation"]) for r in rows]
    e = [float(r["energy_Ha"]) for r in rows]
    err = [max(float(r["abs_error_mHa"]), 1e-12) for r in rows]
    e_fci = summary["convergence_E_fci_Ha"]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True,
                                       gridspec_kw={"height_ratios": [1.3, 1]})
        ax1.plot(it, e, color=NVIDIA_GREEN, lw=1.8, marker="o", ms=3, label="VQE energy")
        ax1.axhline(e_fci, color=FCI_COLOR, ls="--", lw=1.2, label=f"FCI ({e_fci:.6f} Ha)")
        ax1.axhline(summary["convergence_E_hf_Ha"], color=HF_COLOR, ls=":", lw=1.2,
                    label=f"Hartree-Fock ({summary['convergence_E_hf_Ha']:.6f} Ha)")
        ax1.set_ylabel("Energy [Ha]")
        ax1.set_title(f"UCCSD-VQE convergence for H$_2$ at R = {summary['convergence_R_angstrom']} Å")
        ax1.legend(fontsize=9)

        ax2.semilogy(it, err, color=NVIDIA_GREEN, lw=1.8)
        ax2.axhspan(min(err) / 10, CHEMICAL_ACCURACY_MHA, color=NVIDIA_GREEN, alpha=0.12,
                    label="chemical accuracy (< 1.6 mHa)")
        ax2.axhline(CHEMICAL_ACCURACY_MHA, color=NVIDIA_GREEN, lw=1, ls=":")
        ax2.set_ylim(min(err) / 10, max(err) * 3)
        ax2.set_xlabel("Optimizer evaluation")
        ax2.set_ylabel("|E − E$_{FCI}$| [mHa]")
        ax2.legend(fontsize=9, loc="upper right")
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_h2_pes(rows: list[dict], summary: dict, out_path: Path) -> Path:
    """Potential energy curve: VQE points, FCI line, HF dashed, dissociation limit."""
    r = [float(x["R"]) for x in rows]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.plot(r, [float(x["E_fci"]) for x in rows], color=FCI_COLOR, lw=1.6, label="FCI (exact, PySCF)")
        ax.plot(r, [float(x["E_hf"]) for x in rows], color=HF_COLOR, lw=1.4, ls="--", label="Hartree-Fock")
        ax.plot(r, [float(x["E_vqe"]) for x in rows], ls="none", marker="o", ms=5, mfc="white",
                mec=NVIDIA_GREEN, mew=1.6, label="VQE (UCCSD, CUDA-Q)")
        limit = summary["dissociation_limit_2E_H_Ha"]
        ax.axhline(limit, color="0.45", lw=1, ls=":", label=f"2 × E(H atom) = {limit:.4f} Ha")
        ax.plot(summary["R_eq_angstrom"], summary["E_min_vqe_Ha"], marker="*", ms=13, color=NVIDIA_GREEN,
                mec="black", mew=0.6, ls="none",
                label=f"VQE minimum: {summary['R_eq_angstrom']:.3f} Å, {summary['E_min_vqe_Ha']:.4f} Ha")
        ax.set_xlabel("H–H bond length R [Å]")
        ax.set_ylabel("Energy [Ha]")
        ax.set_title("H$_2$ potential energy curve (STO-3G)")
        ax.legend(fontsize=9, loc="upper right")
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_h2_error(rows: list[dict], out_path: Path) -> Path:
    """|E_vqe − E_fci| in mHa vs R on a log axis, with the chemical-accuracy band."""
    r = [float(x["R"]) for x in rows]
    err = [max(float(x["abs_error_mHa"]), 1e-12) for x in rows]
    bottom = min(err) / 10
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.axhspan(bottom, CHEMICAL_ACCURACY_MHA, color=NVIDIA_GREEN, alpha=0.12, label="chemical accuracy")
        ax.axhline(CHEMICAL_ACCURACY_MHA, color=NVIDIA_GREEN, lw=1, ls=":")
        ax.semilogy(r, err, marker="o", ms=4, lw=1.4, color=NVIDIA_GREEN, label="|E$_{VQE}$ − E$_{FCI}$|")
        ax.set_ylim(bottom, CHEMICAL_ACCURACY_MHA * 20)
        ax.text(r[0], CHEMICAL_ACCURACY_MHA * 1.4, "1.6 mHa", fontsize=9, color=NVIDIA_GREEN)
        ax.set_xlabel("H–H bond length R [Å]")
        ax.set_ylabel("|E$_{VQE}$ − E$_{FCI}$| [mHa]")
        ax.set_title(f"VQE error along the curve (max {max(err):.1e} mHa)")
        ax.legend(fontsize=9, loc="upper right")
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_vqe_results(out_dir: Path) -> list[Path]:
    """Write vqe_convergence.png, h2_pes.png and h2_error.png from the result files."""
    out_dir = Path(out_dir)
    summary = json.loads((out_dir / "h2_summary.json").read_text())
    rows = _read_csv(out_dir / "h2_pes.csv")
    return [
        plot_vqe_convergence(out_dir / "vqe_convergence.csv", summary, out_dir / "vqe_convergence.png"),
        plot_h2_pes(rows, summary, out_dir / "h2_pes.png"),
        plot_h2_error(rows, out_dir / "h2_error.png"),
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate plots from result CSVs.")
    sub = parser.add_subparsers(dest="kind", required=True)
    ghz_parser = sub.add_parser("ghz", help="GHZ benchmark plots")
    ghz_parser.add_argument("csv", nargs="*", type=Path,
                            help="GHZ CSVs; default: every <outdir>/ghz_benchmark_*.csv")
    ghz_parser.add_argument("--outdir", type=Path, default=Path("results"))
    ghz_parser.add_argument("--merge-to", type=Path, default=None,
                            help="also write the concatenated rows to this CSV")
    vqe_parser = sub.add_parser("vqe", help="H2 VQE plots")
    vqe_parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    if args.kind == "ghz" and not args.csv:
        args.csv = sorted(p for p in args.outdir.glob("ghz_benchmark_*.csv"))
        if not args.csv:
            parser.error(f"no ghz_benchmark_*.csv files in {args.outdir}")
    if args.kind == "ghz" and args.merge_to is not None:
        merged = merge_ghz_csvs(args.csv, args.merge_to)
        print(f"Merged {len(args.csv)} file(s) into {merged}")
    paths = plot_ghz_results(args.csv, args.outdir) if args.kind == "ghz" else plot_vqe_results(args.outdir)
    for path in paths:
        print(f"Plot written to {path}")


if __name__ == "__main__":
    main()
