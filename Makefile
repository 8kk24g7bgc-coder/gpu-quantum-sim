# gpu-quantum-sim: one entry point for every step.
#
#   make install        install the pinned dependencies
#   make test           run the test suite
#   make quick          tests + reduced experiments (< 2 min each on a CPU)
#   make ghz            GHZ benchmark on this machine -> results/ghz_benchmark_$(GHZ_TAG).csv
#   make vqe            full H2 VQE sweep             -> results/h2_*.csv/json/png
#   make plots          merge all GHZ CSVs and regenerate every plot
#   make all            install, test, both experiments and all plots
#
# Every machine writes its own GHZ file (ghz_benchmark_<tag>.csv); `make plots`
# merges them into results/ghz_benchmark.csv, so a CPU run on a cluster node and
# a GPU run on Colab end up in the same table and figures.
# Examples:
#   make ghz GHZ_TARGETS=nvidia GHZ_TAG=gpu_t4                        # Colab T4
#   make ghz GHZ_TARGETS=qpp-cpu GHZ_TAG=cpu_n24 GHZ_ARGS="--max-qubits 32"

PYTHON      ?= python
RESULTS     ?= results
GHZ_TARGETS ?= qpp-cpu,nvidia
GHZ_TAG     ?= local
GHZ_ARGS    ?=
VQE_ARGS    ?=

.PHONY: help install test quick ghz ghz-quick vqe vqe-quick plots all clean

help:
	@sed -n '3,10p' Makefile | sed 's/^# \{0,1\}//'

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

# Reduced runs go to results/quick/ so they never overwrite real results.
ghz-quick:
	$(PYTHON) -m src.ghz_benchmark --quick --targets $(GHZ_TARGETS) --output $(RESULTS)/quick/ghz_benchmark.csv

vqe-quick:
	$(PYTHON) -m src.h2_vqe --quick --outdir $(RESULTS)/quick

quick: test ghz-quick vqe-quick

ghz:
	$(PYTHON) -m src.ghz_benchmark --targets $(GHZ_TARGETS) $(GHZ_ARGS) \
	    --output $(RESULTS)/ghz_benchmark_$(GHZ_TAG).csv --no-plot

vqe:
	$(PYTHON) -m src.h2_vqe $(VQE_ARGS) --outdir $(RESULTS)

plots:
	$(PYTHON) -m src.plotting ghz --merge-to $(RESULTS)/ghz_benchmark.csv --outdir $(RESULTS)
	$(PYTHON) -m src.plotting vqe --outdir $(RESULTS)

all: install test ghz vqe plots

clean:
	rm -rf $(RESULTS)/quick .pytest_cache src/__pycache__ tests/__pycache__
