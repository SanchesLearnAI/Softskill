#!/bin/bash
#SBATCH --job-name=hybrid-beh-hard-s123
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hybrid_behavior_task_hard_s123_%j.out
#SBATCH --error=logs/hybrid_behavior_task_hard_s123_%j.err

set -euo pipefail
cd "${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"

for seed in 1 2 3; do
  SEED="${seed}" \
  OUTPUT_DIR="outputs/behavior_compression/v1/hybrid_shared_behavior_task_hard_two_stage/seed${seed}" \
  bash scripts/experiments/train_hybrid_shared_behavior_task_hard.sh
done
