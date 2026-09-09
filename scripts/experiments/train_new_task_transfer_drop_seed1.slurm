#!/bin/bash
#SBATCH --job-name=drop-transfer-s1
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/drop_transfer_seed1_%j.out
#SBATCH --error=logs/drop_transfer_seed1_%j.err

set -uo pipefail

ROOT="${SOFTSKILL_WORKSPACE_ROOT:?Set SOFTSKILL_WORKSPACE_ROOT to the directory containing SoftSkill}"
PROJECT="$ROOT/SoftSkill"
MODEL_SNAPSHOT="$ROOT/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATA_ROOT="$PROJECT/data/drop"
OUTPUT_ROOT="$PROJECT/outputs/new_task_transfer/v1/drop"
SOURCE_CHECKPOINT="$PROJECT/outputs/behavior_compression/v1/shared16_task16_raw_two_stage/seed1/best_v2.pt"

source "$ROOT/activate_softskill.sh"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for required in \
  "$PROJECT/skillopt/behavior_compression/new_task_transfer/v1/tokenizer_audit.json" \
  "$PROJECT/skillopt/behavior_compression/v1/shared_behavior.md" \
  "$SOURCE_CHECKPOINT" \
  "$MODEL_SNAPSHOT/config.json" \
  "$DATA_ROOT/raw/drop_dataset/drop_dataset_train.json" \
  "$DATA_ROOT/raw/drop_dataset/drop_dataset_dev.json"; do
  if [[ ! -f "$required" ]]; then
    echo "bundle=drop_seed1 status=failed missing_required_file=$required" >&2
    exit 91
  fi
done

run_condition() {
  local condition="$1"
  local output_dir="$OUTPUT_ROOT/$condition/seed1"
  local started ended exit_code
  if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "condition=$condition refusal=nonempty_output output_dir=$output_dir" >&2
    return 90
  fi
  started="$(date --iso-8601=seconds)"
  echo "condition=$condition event=start time=$started output_dir=$output_dir"
  local source_args=()
  if [[ "$condition" == "f1" ]]; then
    source_args=(--source_checkpoint "$SOURCE_CHECKPOINT")
  fi
  if python -m skillopt.behavior_compression.new_task_transfer.v1.train \
    --task drop \
    --condition "$condition" \
    --seed 1 \
    --output_dir "$output_dir" \
    --max_optimizer_steps 150 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --batch_size 1 \
    --model_name "$MODEL_SNAPSHOT" \
    --data_root "$DATA_ROOT" \
    --max_prompt_tokens 8192 \
    --max_target_tokens 128 \
    --eval_batch_size 1 \
    --checkpoint_every_steps 50 \
    "${source_args[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi
  ended="$(date --iso-8601=seconds)"
  echo "condition=$condition event=exit exit_code=$exit_code start=$started end=$ended"
  if [[ $exit_code -ne 0 ]]; then
    return "$exit_code"
  fi
  if python - "$output_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("status") != "complete":
    raise SystemExit(f"missing complete summary: {path}")
PY
  then
    return 0
  else
    exit_code=$?
    echo "condition=$condition event=verification_failed exit_code=$exit_code" >&2
    return "$exit_code"
  fi
}

for condition in f0 f1 i32; do
  if run_condition "$condition"; then
    :
  else
    exit_code=$?
    echo "bundle=drop_seed1 status=failed condition=$condition exit_code=$exit_code" >&2
    exit "$exit_code"
  fi
done

echo "bundle=drop_seed1 status=complete"
