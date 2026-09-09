#!/usr/bin/env bash
set -euo pipefail

: "${CELL:?Set CELL=A_direct or C_direct_shared}"
SEED="${SEED:-1}"

case "${CELL}" in
  A_direct)
    LAYOUT=independent_2x16
    DEFAULT_OUTPUT_DIR="outputs/behavior_compression/v1/uniform/A_direct/seed${SEED}"
    ;;
  C_direct_shared)
    LAYOUT=shared16_task16
    DEFAULT_OUTPUT_DIR="outputs/behavior_compression/v1/uniform/C_direct_shared/seed${SEED}"
    ;;
  *)
    echo "Unknown CELL=${CELL}; expected A_direct or C_direct_shared" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  echo "Skipping completed behavior-compression run: ${OUTPUT_DIR}"
  exit 0
fi

python scripts/train_joint_soft_prefix_v2.py \
  --model_name "${MODEL_NAME:-Qwen/Qwen3.5-4B}" \
  --out_root "${OUTPUT_DIR}" \
  --layout "${LAYOUT}" \
  --initialization_mode behavior_markdown \
  --shared_behavior_path skillopt/behavior_compression/v1/shared_behavior.md \
  --behavior_provenance_path skillopt/behavior_compression/v1/provenance.json \
  --behavior_tokenizer_audit_path skillopt/behavior_compression/v1/tokenizer_audit.json \
  --seed "${SEED}" \
  --prefix_length 16 \
  --num_epochs 3 \
  --expected_optimizer_steps 150 \
  --learning_rate 0.001 \
  --lr_schedule uniform
