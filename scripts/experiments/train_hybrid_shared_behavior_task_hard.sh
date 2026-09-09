#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/behavior_compression/v1/hybrid_shared_behavior_task_hard_two_stage/seed${SEED}}"

if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  echo "Skipping completed hybrid behavior/Hard-Markdown run: ${OUTPUT_DIR}"
  exit 0
fi

python scripts/train_joint_soft_prefix_v2.py \
  --model_name "${MODEL_NAME:-Qwen/Qwen3.5-4B}" \
  --out_root "${OUTPUT_DIR}" \
  --layout shared16_task16 \
  --initialization_mode hybrid_shared_behavior_task_hard \
  --shared_behavior_path skillopt/behavior_compression/v1/shared_behavior.md \
  --behavior_provenance_path skillopt/behavior_compression/v1/provenance.json \
  --behavior_tokenizer_audit_path skillopt/behavior_compression/v1/tokenizer_audit.json \
  --seed "${SEED}" \
  --prefix_length 16 \
  --num_epochs 3 \
  --expected_optimizer_steps 150 \
  --lr_schedule behavior_two_stage \
  --behavior_warmup_fraction 0.2 \
  --behavior_shared_warmup_lr 0.001 \
  --behavior_shared_joint_lr 0.0001 \
  --behavior_task_joint_lr 0.001 \
  --learning_rate 0.001
