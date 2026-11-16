#!/usr/bin/env bash
# Same steps as `make all`, for machines without `make` (e.g. HPC compute nodes).
#
#   ./run_all.sh                    install, test, GHZ benchmark, H2 VQE, all plots
#   ./run_all.sh --quick            install, test and the reduced runs (results/quick/)
#   ./run_all.sh --skip-install     do not run pip (environment already set up)
#
# Environment variables (same meaning as in the Makefile):
#   PYTHON=python  GHZ_TARGETS=qpp-cpu,nvidia  GHZ_TAG=local  GHZ_ARGS=""
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
RESULTS=${RESULTS:-results}
GHZ_TARGETS=${GHZ_TARGETS:-qpp-cpu,nvidia}
GHZ_TAG=${GHZ_TAG:-local}
GHZ_ARGS=${GHZ_ARGS:-}

quick=0
install=1
for arg in "$@"; do
  case "$arg" in
    --quick) quick=1 ;;
    --skip-install) install=0 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

step() { echo; echo "==> $*"; }

if [ "$install" -eq 1 ]; then
  step "Installing pinned dependencies"
  "$PYTHON" -m pip install -r requirements.txt
fi

step "Running tests"
"$PYTHON" -m pytest -q

if [ "$quick" -eq 1 ]; then
  step "GHZ benchmark (quick) -> $RESULTS/quick/"
  "$PYTHON" -m src.ghz_benchmark --quick --targets "$GHZ_TARGETS" --output "$RESULTS/quick/ghz_benchmark.csv"
  step "H2 VQE (quick) -> $RESULTS/quick/"
  "$PYTHON" -m src.h2_vqe --quick --outdir "$RESULTS/quick"
  exit 0
fi

step "GHZ benchmark -> $RESULTS/ghz_benchmark_${GHZ_TAG}.csv"
# shellcheck disable=SC2086  # GHZ_ARGS is intentionally word-split
"$PYTHON" -m src.ghz_benchmark --targets "$GHZ_TARGETS" $GHZ_ARGS \
    --output "$RESULTS/ghz_benchmark_${GHZ_TAG}.csv" --no-plot

step "H2 VQE sweep -> $RESULTS/"
"$PYTHON" -m src.h2_vqe --outdir "$RESULTS"

step "Merging GHZ results and regenerating plots"
"$PYTHON" -m src.plotting ghz --merge-to "$RESULTS/ghz_benchmark.csv" --outdir "$RESULTS"
"$PYTHON" -m src.plotting vqe --outdir "$RESULTS"

step "Regenerating README.md from the results"
"$PYTHON" scripts/build_readme.py
