#!/bin/bash
#
# GHZ statevector benchmark on the CPU (CUDA-Q qpp-cpu), n = 20..32 qubits.
# Submit from the cluster/ folder:   cd ~/gpu-quantum-sim/cluster && sbatch sbatch_ghz_cpu_n24.sh
#
#SBATCH --job-name=ghz_cpu
#SBATCH --output=slurm_ghz_cpu_%j.txt
#SBATCH --partition=cpu1
#SBATCH --nodelist=n24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nice=200
#SBATCH --time=2-12:00:00

# 1. Define directory names [DO NOT CHANGE]
# =========================================

# get name of the temporary directory working directory,
#physically on the compute-node in the folder /scratch where everyone has access.

workdir=/scratch/c_qsim/job_${SLURM_JOB_ID}

# get submit directory
# (every file/folder below this directory is copied to the compute node)
submitdir=${SLURM_SUBMIT_DIR}


# 2. Transfer to node [DO NOT CHANGE]
# ===================================

# create/empty the temporary directory on the compute node
if [ ! -d "${workdir}" ]; then
  mkdir -p "${workdir}"
else
  rm -rf "${workdir}"/*
fi

# change current directory to the location of the sbatch command
# ("submitdir" is somewhere in the home directory on the head node)
cd "${submitdir}"
# copy all files/folders in "submitdir" to "workdir"
# ("workdir" == temporary directory on the compute node)
cp -prf * ${workdir}
# change directory to the temporary directory on the compute-node
cd ${workdir}


# 3. Function to transfer back to the head node [DO NOT CHANGE]
# =============================================================

# define clean-up function
function clean_up {
  # - remove log-file on the compute-node, to avoid the one created by SLURM is overwritten
  rm job.slurm.out
  # - delete temporary files from the compute-node, before copying
  # rm -r ...
  # - change directory to the location of the sbatch command (on the head node)
  cd "${submitdir}"
  # - copy everything from the temporary directory on the compute-node
  cp -prf "${workdir}"/* .
  # - erase the temporary directory from the compute-node
  rm -rf "${workdir}"/*
  rm -rf "${workdir}"
  # - exit the script
  exit
}

# call "clean_up" function when this script exits, it is run even if SLURM cancels the job
trap 'clean_up' EXIT


# 4. Execute
# ==========
# The CSV and the log are written straight to the home directory (NFS), not to
# /scratch: the benchmark saves a row after every qubit count, so if the node
# fails during the ~30 h run, every finished measurement is already safe.

REPO=${HOME}/gpu-quantum-sim
outfile=${submitdir}/output.${SLURM_JOB_ID}

source ${HOME}/venvs/gpuqsim/bin/activate
export PYTHONPATH=${REPO}

# one OpenMP thread per physical core (no hyperthreads): statevector updates
# are memory-bound, so extra hyperthreads only compete for bandwidth
PHYS_CORES=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
export OMP_NUM_THREADS=${PHYS_CORES}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

{
  echo "job=${SLURM_JOB_ID}  host=$(hostname)  start=$(date '+%F %T')"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}  (git commit: see \"git_commit\" in the hardware block)"
  python -u -m src.ghz_benchmark \
      --targets qpp-cpu \
      --min-qubits 20 --max-qubits 32 \
      --output ${REPO}/results/ghz_benchmark_cpu_n24.csv \
      --no-plot
  rc=$?
  echo "end=$(date '+%F %T')  exit_code=${rc}"
} >> ${outfile} 2>&1
