#!/usr/bin/env bash
set -euo pipefail

: "${SEED:?Set SEED=1|2|3}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/four_cell/C_shared16_task16_direct/seed${SEED}}"
if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  echo "Skipping completed run: ${OUTPUT_DIR}"
  exit 0
fi

python scripts/train_shared_task_soft_prefix.py \
  --model_name "${MODEL_NAME:-Qwen/Qwen3.5-4B}" \
  --out_root "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --prefix_length 16 \
  --num_epochs "${NUM_EPOCHS:-3}" \
  --shared_lr_start "${SHARED_LR_START:-0.001}" \
  --shared_lr_min "${SHARED_LR_MIN:-0.00005}" \
  --shared_decay "${SHARED_DECAY:-3.0}" \
  --task_lr_max "${TASK_LR_MAX:-0.001}" \
  --task_growth "${TASK_GROWTH:-5.0}" \
  --disable_residual_reparameterization
