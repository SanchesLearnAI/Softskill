#!/usr/bin/env bash
set -euo pipefail

: "${FAMILY:?Set FAMILY=A_direct|B_residual|C_direct_shared|D_residual_shared}"
: "${SEED:?Set SEED=1|2|3}"

case "${FAMILY}" in
  A_direct)
    LAYOUT="independent_2x16"
    CELL="A_joint_2x16_direct"
    RESIDUAL=()
    ;;
  B_residual)
    LAYOUT="independent_2x16"
    CELL="B_joint_2x16_residual"
    RESIDUAL=(--residual_reparameterization)
    ;;
  C_direct_shared)
    LAYOUT="shared16_task16"
    CELL="C_shared16_task16_direct"
    RESIDUAL=()
    ;;
  D_residual_shared)
    LAYOUT="shared16_task16"
    CELL="D_shared16_task16_residual"
    RESIDUAL=(--residual_reparameterization)
    ;;
  *)
    echo "Unknown FAMILY=${FAMILY}" >&2
    exit 2
    ;;
esac

LR_MODE="${LR_SCHEDULE:-uniform}"
INIT_MODE="${INITIALIZATION_MODE:-legacy_task_blocks}"
DEFAULT_RESIDUAL_MODE="global_matched"
if [[ "${LR_MODE}" == "progressive" && "${FAMILY}" == "D_residual_shared" ]]; then
  DEFAULT_RESIDUAL_MODE="branch_decoupled"
fi
ACTIVE_RESIDUAL_MODE="${RESIDUAL_MODE:-${DEFAULT_RESIDUAL_MODE}}"
if [[ "${INIT_MODE}" == "behavior_markdown" && "${LR_MODE}" == "uniform" ]]; then
  DEFAULT_OUTPUT_DIR="outputs/behavior_compression/v1/${CELL}/seed${SEED}"
elif [[ "${INIT_MODE}" == "behavior_markdown" ]]; then
  DEFAULT_OUTPUT_DIR="outputs/behavior_compression/v1/progressive/${CELL}_${ACTIVE_RESIDUAL_MODE}/seed${SEED}"
elif [[ "${LR_MODE}" == "uniform" ]]; then
  DEFAULT_OUTPUT_DIR="outputs/v2/four_cell/${CELL}/seed${SEED}"
else
  DEFAULT_OUTPUT_DIR="outputs/v2/progressive/${CELL}_${ACTIVE_RESIDUAL_MODE}/seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  echo "Skipping completed v2 run: ${OUTPUT_DIR}"
  exit 0
fi

COMMAND=(
  python scripts/train_joint_soft_prefix_v2.py
  --model_name "${MODEL_NAME:-Qwen/Qwen3.5-4B}"
  --out_root "${OUTPUT_DIR}"
  --layout "${LAYOUT}"
  --initialization_mode "${INIT_MODE}"
  --shared_behavior_path "${SHARED_BEHAVIOR_PATH:-skillopt/behavior_compression/v1/shared_behavior.md}"
  --seed "${SEED}"
  --prefix_length 16
  --num_epochs "${NUM_EPOCHS:-3}"
  --learning_rate "${LEARNING_RATE:-0.001}"
  --lr_schedule "${LR_MODE}"
  --shared_lr_start "${SHARED_LR_START:-0.001}"
  --shared_lr_min "${SHARED_LR_MIN:-0.00005}"
  --shared_decay "${SHARED_DECAY:-3.0}"
  --task_lr_max "${TASK_LR_MAX:-0.001}"
  --task_growth "${TASK_GROWTH:-5.0}"
  --residual_bottleneck_size "${RESIDUAL_BOTTLENECK_SIZE:-400}"
  --residual_mode "${ACTIVE_RESIDUAL_MODE}"
)
if [[ -n "${RESIDUAL_INIT_SEED:-}" ]]; then
  COMMAND+=(--residual_init_seed "${RESIDUAL_INIT_SEED}")
fi
if (( ${#RESIDUAL[@]} > 0 )); then
  COMMAND+=("${RESIDUAL[@]}")
fi
"${COMMAND[@]}"
