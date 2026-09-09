#!/usr/bin/env bash
#SBATCH --job-name=prefix-state-v2
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prefix_state_%j.out
#SBATCH --error=logs/prefix_state_%j.err

set -euo pipefail

PROJECT_ROOT="${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
cd "${PROJECT_ROOT}"
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

INDEPENDENT_STATES="${INDEPENDENT_STATES:-outputs/v2/four_cell/B_joint_2x16_residual/seed1/prefix_states.pt}"
SHARED_STATES="${SHARED_STATES:-outputs/v2/four_cell/D_shared16_task16_residual/seed1/prefix_states.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/v2/prefix_state_trajectory/B_vs_D_seed1}"

python scripts/analyze_prefix_state_trajectory.py \
  --independent-states "${INDEPENDENT_STATES}" \
  --shared-states "${SHARED_STATES}" \
  --output-dir "${OUTPUT_DIR}"
