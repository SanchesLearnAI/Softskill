#!/bin/bash
#SBATCH --job-name=v2-four-cell
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v2_%x_%j.out
#SBATCH --error=logs/v2_%x_%j.err

set -euo pipefail

cd "${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"
mkdir -p logs outputs/v2/four_cell

: "${FAMILY:?Set FAMILY=A_direct|B_residual|C_direct_shared|D_residual_shared}"
SEEDS="${SEEDS:-1}"
failed=0

for seed in ${SEEDS}; do
  echo "Starting v2 family=${FAMILY} seed=${seed} at $(date --iso-8601=seconds)"
  if SEED="${seed}" bash scripts/experiments/train_v2_four_cell.sh; then
    echo "Completed v2 family=${FAMILY} seed=${seed}"
  else
    echo "FAILED v2 family=${FAMILY} seed=${seed}" >&2
    failed=1
  fi
done

exit "${failed}"
