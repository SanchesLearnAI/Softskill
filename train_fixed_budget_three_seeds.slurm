#!/bin/bash
#SBATCH --job-name=fixed-budget-3seeds
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fixed_budget_3seeds_%x_%j.out
#SBATCH --error=logs/fixed_budget_3seeds_%x_%j.err

set -uo pipefail

cd "${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"
mkdir -p logs outputs/fixed_budget

: "${FAMILY:?Set FAMILY=two_by_16|shared16_task16}"
if [[ "${FAMILY}" == "two_by_16" ]]; then
  : "${TASK:?Set TASK=searchqa|livemath|docvqa}"
fi

failed=0
for seed in 1 2 3; do
  echo "============================================================"
  echo "Starting family=${FAMILY} task=${TASK:-joint} seed=${seed}"
  echo "Started at $(date --iso-8601=seconds)"
  echo "============================================================"

  if [[ "${FAMILY}" == "two_by_16" ]]; then
    if TASK="${TASK}" SEED="${seed}" \
      bash scripts/experiments/train_fixed_budget_2x16.sh; then
      echo "Completed task=${TASK} seed=${seed}"
    else
      echo "FAILED task=${TASK} seed=${seed}" >&2
      failed=1
    fi
  elif [[ "${FAMILY}" == "shared16_task16" ]]; then
    if SEED="${seed}" \
      bash scripts/experiments/train_fixed_budget_shared16_task16.sh; then
      echo "Completed Shared16+Task16 seed=${seed}"
    else
      echo "FAILED Shared16+Task16 seed=${seed}" >&2
      failed=1
    fi
  else
    echo "Unknown FAMILY=${FAMILY}" >&2
    exit 2
  fi
done

exit "${failed}"
