#!/bin/bash
#SBATCH --job-name=chartqa-chunked-s1
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/chartqa_chunked_seed1_%j.out
#SBATCH --error=logs/chartqa_chunked_seed1_%j.err

set -uo pipefail

ROOT="${SOFTSKILL_WORKSPACE_ROOT:?Set SOFTSKILL_WORKSPACE_ROOT to the directory containing SoftSkill}"
PROJECT="$ROOT/SoftSkill"
PYTHON="$ROOT/envs/softskill/bin/python"
MODEL_SNAPSHOT="$ROOT/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATA_ROOT="$PROJECT/data/chartqa"
OUTPUT_ROOT="$PROJECT/outputs/new_task_transfer/v1/chartqa"
SOURCE_CHECKPOINT="$PROJECT/outputs/behavior_compression/v1/shared16_task16_raw_two_stage/seed1/best_v2.pt"
CHUNK_MODULE=skillopt.behavior_compression.new_task_transfer.v1.chartqa_chunked_eval
CHUNK_SIZE=256
MAX_MEAN_SAMPLE_SECONDS=3.0
MAX_PEAK_RSS_KB=$((70 * 1024 * 1024))

cd "$PROJECT"
export PATH="$ROOT/envs/softskill/bin:$PATH"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export HF_HOME="$ROOT/hf_cache"
export TRANSFORMERS_CACHE="$ROOT/hf_cache/transformers"
export HF_DATASETS_CACHE="$ROOT/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

for required in \
  "$PYTHON" \
  "$MODEL_SNAPSHOT/config.json" \
  "$SOURCE_CHECKPOINT" \
  "$DATA_ROOT/ChartQA Dataset/train/train_human.json" \
  "$DATA_ROOT/ChartQA Dataset/val/val_human.json" \
  "$DATA_ROOT/ChartQA Dataset/test/test_human.json" \
  "$PROJECT/skillopt/behavior_compression/new_task_transfer/v1/chartqa_chunked_eval.py"; do
  if [[ ! -f "$required" ]]; then
    echo "bundle=chartqa_chunked_seed1 status=failed missing_required_file=$required" >&2
    exit 91
  fi
done

run_condition() {
  local condition="$1"
  local output_dir="$OUTPUT_ROOT/$condition/seed1"
  local final_checkpoint="$output_dir/final.pt"
  local resume_args=()
  local source_args=()
  local exit_code

  if [[ "$condition" == "f1" ]]; then
    source_args=(--source_checkpoint "$SOURCE_CHECKPOINT")
  fi
  if [[ -s "$output_dir/summary.json" ]]; then
    if "$PYTHON" - "$output_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if json.loads(path.read_text(encoding="utf-8")).get("status") != "complete":
    raise SystemExit(f"incomplete existing summary: {path}")
PY
    then
      echo "condition=$condition event=skip_complete output_dir=$output_dir"
      return 0
    else
      return 92
    fi
  fi
  if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    if [[ -s "$output_dir/step0_audit.json" && -s "$output_dir/transfer_sanity.json" && -s "$output_dir/latest.pt" ]]; then
      resume_args=(--resume)
    else
      echo "condition=$condition refusal=nonresumable_output output_dir=$output_dir" >&2
      return 90
    fi
  fi

  echo "condition=$condition phase=train_to_final event=start time=$(date --iso-8601=seconds)"
  if timeout --signal=TERM --kill-after=30s 15m \
    "$PYTHON" -m skillopt.behavior_compression.new_task_transfer.v1.train \
    --task chartqa \
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
    --checkpoint_every_steps 50 \
    --train_only \
    "${resume_args[@]}" \
    "${source_args[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi
  echo "condition=$condition phase=train_to_final event=exit exit_code=$exit_code time=$(date --iso-8601=seconds)"
  if [[ $exit_code -ne 0 ]]; then
    return "$exit_code"
  fi
  if [[ ! -s "$final_checkpoint" ]]; then
    echo "condition=$condition status=failed missing_final_checkpoint=$final_checkpoint" >&2
    return 95
  fi

  local split total_samples source_split formal_path start end chunk_index manifest
  for split in validation test; do
    if [[ "$split" == "validation" ]]; then
      total_samples=960
      source_split=val
    else
      total_samples=1250
      source_split=test
    fi
    formal_path="$output_dir/predictions/$split.jsonl"
    if [[ -s "$formal_path" ]]; then
      echo "condition=$condition split=$split event=reuse_complete_formal_predictions"
      continue
    fi

    chunk_index=0
    for ((start=0; start<total_samples; start+=CHUNK_SIZE)); do
      end=$((start + CHUNK_SIZE))
      if ((end > total_samples)); then
        end=$total_samples
      fi
      echo "condition=$condition split=$split phase=chunk event=start index=$chunk_index start=$start end=$end time=$(date --iso-8601=seconds)"
      if timeout --signal=TERM --kill-after=30s 15m \
        "$PYTHON" -m "$CHUNK_MODULE" worker \
        --model_name "$MODEL_SNAPSHOT" \
        --data_root "$DATA_ROOT" \
        --output_dir "$output_dir" \
        --checkpoint "$final_checkpoint" \
        --condition "$condition" \
        --seed 1 \
        --split "$split" \
        --start "$start" \
        --chunk_size "$CHUNK_SIZE" \
        --max_prompt_tokens 8192 \
        --max_new_tokens 64 \
        --max_image_tokens 0 \
        "${source_args[@]}"; then
        exit_code=0
      else
        exit_code=$?
      fi
      echo "condition=$condition split=$split phase=chunk event=exit index=$chunk_index exit_code=$exit_code time=$(date --iso-8601=seconds)"
      if [[ $exit_code -ne 0 ]]; then
        return "$exit_code"
      fi

      if [[ $chunk_index -eq 0 ]]; then
        manifest="$output_dir/predictions/chunks/$split/${split}_$(printf '%06d' "$start")_$(printf '%06d' "$end").manifest.json"
        if ! "$PYTHON" - "$manifest" "$MAX_MEAN_SAMPLE_SECONDS" "$MAX_PEAK_RSS_KB" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
max_mean = float(sys.argv[2])
max_rss = int(sys.argv[3])
progress = payload.get("progress", {})
recorded = int(progress.get("recorded_samples", 0))
expected = int(payload.get("sample_count", -1))
mean = progress.get("mean_sample_seconds")
peak_rss = progress.get("peak_vmrss_kb")
if recorded != expected or mean is None:
    raise SystemExit(f"first chunk has incomplete progress evidence: {recorded}/{expected}")
if float(mean) > max_mean:
    raise SystemExit(f"first chunk mean {mean:.4f}s exceeds {max_mean:.4f}s")
if peak_rss is not None and int(peak_rss) > max_rss:
    raise SystemExit(f"first chunk peak RSS {peak_rss} KiB exceeds {max_rss} KiB")
print(f"first_chunk_gate=passed mean_sample_seconds={float(mean):.6f} peak_vmrss_kb={peak_rss}")
PY
        then
          return 94
        fi
      fi
      chunk_index=$((chunk_index + 1))
    done

    echo "condition=$condition split=$split phase=merge event=start time=$(date --iso-8601=seconds)"
    if timeout --signal=TERM --kill-after=30s 5m \
      "$PYTHON" -m "$CHUNK_MODULE" merge \
      --data_root "$DATA_ROOT" \
      --output_dir "$output_dir" \
      --checkpoint "$final_checkpoint" \
      --condition "$condition" \
      --seed 1 \
      --split "$split" \
      --chunk_size "$CHUNK_SIZE" \
      --max_prompt_tokens 8192 \
      --max_new_tokens 64 \
      --max_image_tokens 0 \
      "${source_args[@]}"; then
      echo "condition=$condition split=$split phase=merge event=complete time=$(date --iso-8601=seconds)"
    else
      exit_code=$?
      return "$exit_code"
    fi
  done

  echo "condition=$condition phase=finalize_summary event=start time=$(date --iso-8601=seconds)"
  if timeout --signal=TERM --kill-after=30s 15m \
    "$PYTHON" -m skillopt.behavior_compression.new_task_transfer.v1.train \
    --task chartqa \
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
    --resume \
    "${source_args[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi
  if [[ $exit_code -ne 0 ]]; then
    return "$exit_code"
  fi
  if ! "$PYTHON" - "$output_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "complete" or int(payload.get("optimizer_steps", -1)) != 150:
    raise SystemExit(f"missing complete summary: {path}")
PY
  then
    return 96
  fi
  echo "condition=$condition event=complete time=$(date --iso-8601=seconds)"
}

for condition in f0 f1 i32; do
  if run_condition "$condition"; then
    :
  else
    exit_code=$?
    echo "bundle=chartqa_chunked_seed1 status=failed condition=$condition exit_code=$exit_code" >&2
    exit "$exit_code"
  fi
done

echo "bundle=chartqa_chunked_seed1 status=complete time=$(date --iso-8601=seconds)"
