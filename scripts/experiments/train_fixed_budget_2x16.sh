#!/usr/bin/env bash
set -euo pipefail

: "${TASK:?Set TASK=searchqa|livemath|docvqa}"
: "${SEED:?Set SEED=1|2|3}"

case "${TASK}" in
  searchqa)
    CONFIG="configs/searchqa/soft_prefix_2x16.yaml"
    SPLIT_DIR="data/searchqa_split"
    ;;
  livemath)
    CONFIG="configs/livemathematicianbench/soft_prefix_2x16.yaml"
    SPLIT_DIR="data/livemathematicianbench_split"
    ;;
  docvqa)
    CONFIG="configs/docvqa/soft_prefix_2x16.yaml"
    SPLIT_DIR="data/docvqa/splits"
    ;;
  *)
    echo "Unknown TASK=${TASK}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-outputs/fixed_budget/two_by_16/${TASK}_seed${SEED}}"
if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  echo "Skipping completed run: ${OUTPUT_DIR}"
  exit 0
fi

python scripts/train_soft_prefix.py \
  --config "${CONFIG}" \
  --split_dir "${SPLIT_DIR}" \
  --model_name "${MODEL_NAME:-Qwen/Qwen3.5-4B}" \
  --cfg-options \
    "train.seed=${SEED}" \
    "env.split_seed=${SEED}" \
    "soft_prefix.num_soft_skills=2" \
    "soft_prefix.prefix_length=16" \
    "soft_prefix.init_strategy=text" \
    "soft_prefix.inference_backend=${INFERENCE_BACKEND:-local_hf}" \
  --out_root "${OUTPUT_DIR}"
