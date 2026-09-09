#!/bin/bash
#SBATCH --job-name=behavior-v1-pair
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/behavior_v1_pair_%j.out
#SBATCH --error=logs/behavior_v1_pair_%j.err

set -euo pipefail
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"

: "${CELL:?Submit with --export=ALL,CELL=A_direct or C_direct_shared}"
SEED="${SEED:-1}" bash scripts/experiments/train_behavior_compression_v1_pair.sh
