#!/bin/bash
#SBATCH --job-name=hybrid-rpt-s123
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hybrid_residual_global_s123_%j.out
#SBATCH --error=logs/hybrid_residual_global_s123_%j.err

set -euo pipefail
PROJECT_ROOT="${RESIDUAL_PROJECT_ROOT:-${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT}}"
OUTPUT_ROOT="${RESIDUAL_OUTPUT_ROOT:-outputs/behavior_compression/v1/hybrid_shared_behavior_task_hard_residual_global_two_stage}"
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"
cd "${PROJECT_ROOT}"

for seed in 1 2 3; do
  SEED="${seed}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/seed${seed}" \
  bash scripts/experiments/train_hybrid_shared_behavior_task_hard_residual.sh
done
