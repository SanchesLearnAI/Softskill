#!/bin/bash
#SBATCH --job-name=behavior-v1-smoke
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/behavior_v1_smoke_%j.out
#SBATCH --error=logs/behavior_v1_smoke_%j.err

set -euo pipefail
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"

python scripts/smoke_joint_soft_prefix_v2.py \
  --layout shared16_task16 \
  --initialization_mode behavior_markdown \
  --shared_behavior_path skillopt/behavior_compression/v1/shared_behavior.md \
  --behavior_two_stage \
  --seed 1 \
  --out_root outputs/behavior_compression/v1/smoke/seed1
