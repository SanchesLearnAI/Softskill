#!/bin/bash
#SBATCH --job-name=prefix-sim-v2
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prefix_similarity_%j.out
#SBATCH --error=logs/prefix_similarity_%j.err

set -euo pipefail

cd "${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

INDEPENDENT_ROOT="${INDEPENDENT_ROOT:-outputs/v2/four_cell/B_joint_2x16_residual/seed1}"
SHARED_ROOT="${SHARED_ROOT:-outputs/v2/four_cell/D_shared16_task16_residual/seed1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/v2/prefix_similarity/B_vs_D_seed1}"
PREFIX_LENGTH="${PREFIX_LENGTH:-16}"

python scripts/analyze_prefix_similarity.py \
  --independent-root "${INDEPENDENT_ROOT}" \
  --shared-root "${SHARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --prefix-length "${PREFIX_LENGTH}"
