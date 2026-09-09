#!/bin/bash
#SBATCH --job-name=four-cell-3seeds
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/four_cell_%x_%j.out
#SBATCH --error=logs/four_cell_%x_%j.err

set -uo pipefail

cd "${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"
mkdir -p logs outputs/fixed_budget outputs/four_cell

: "${FAMILY:?Set FAMILY=A_direct|B_residual|C_direct_shared|D_residual_shared}"
failed=0

for seed in 1 2 3; do
  echo "============================================================"
  echo "Starting family=${FAMILY} seed=${seed} at $(date --iso-8601=seconds)"
  echo "============================================================"

  case "${FAMILY}" in
    A_direct)
      SEED="${seed}" bash scripts/experiments/train_fixed_budget_joint_2x16_direct.sh
      ;;
    B_residual)
      SEED="${seed}" bash scripts/experiments/train_fixed_budget_joint_2x16.sh
      ;;
    C_direct_shared)
      SEED="${seed}" bash scripts/experiments/train_fixed_budget_shared16_task16_direct.sh
      ;;
    D_residual_shared)
      SEED="${seed}" bash scripts/experiments/train_fixed_budget_shared16_task16.sh
      ;;
    *)
      echo "Unknown FAMILY=${FAMILY}" >&2
      exit 2
      ;;
  esac

  status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "FAILED family=${FAMILY} seed=${seed}" >&2
    failed=1
  else
    echo "Completed family=${FAMILY} seed=${seed}"
  fi
done

exit "${failed}"
