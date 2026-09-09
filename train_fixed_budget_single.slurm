#!/bin/bash
#SBATCH --job-name=fixed-budget
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fixed_budget_%x_%j.out
#SBATCH --error=logs/fixed_budget_%x_%j.err

set -euo pipefail

cd "${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"
mkdir -p logs outputs/fixed_budget

: "${FAMILY:?Set FAMILY=two_by_16|shared16_task16}"
: "${SEED:?Set SEED=1|2|3}"

case "${FAMILY}" in
  joint_two_by_16_direct)
    exec bash scripts/experiments/train_fixed_budget_joint_2x16_direct.sh
    ;;
  shared16_task16_direct)
    exec bash scripts/experiments/train_fixed_budget_shared16_task16_direct.sh
    ;;
  joint_two_by_16)
    exec bash scripts/experiments/train_fixed_budget_joint_2x16.sh
    ;;
  two_by_16)
    : "${TASK:?Set TASK=searchqa|livemath|docvqa}"
    exec bash scripts/experiments/train_fixed_budget_2x16.sh
    ;;
  shared16_task16)
    exec bash scripts/experiments/train_fixed_budget_shared16_task16.sh
    ;;
  *)
    echo "Unknown FAMILY=${FAMILY}" >&2
    exit 2
    ;;
esac
