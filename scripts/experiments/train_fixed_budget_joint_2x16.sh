#!/usr/bin/env bash
set -euo pipefail

: "${SEED:?Set SEED=1|2|3}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/fixed_budget/joint_two_by_16/seed${SEED}}"
if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  echo "Skipping completed run: ${OUTPUT_DIR}"
  exit 0
fi

python scripts/train_joint_two_prefix.py \
  --model_name "${MODEL_NAME:-Qwen/Qwen3.5-4B}" \
  --out_root "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --prefix_length 16 \
  --num_soft_skills 2 \
  --num_epochs "${NUM_EPOCHS:-3}" \
  --learning_rate "${LEARNING_RATE:-0.001}" \
  --residual_bottleneck_size "${RESIDUAL_BOTTLENECK_SIZE:-400}"
