#!/bin/bash
#SBATCH --job-name=behavior-v1-seed1
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/behavior_v1_seed1_%j.out
#SBATCH --error=logs/behavior_v1_seed1_%j.err

set -euo pipefail
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"

SEED=1 \
OUTPUT_DIR=outputs/behavior_compression/v1/shared16_task16_raw_two_stage/seed1 \
bash scripts/experiments/train_behavior_compression_v1.sh
