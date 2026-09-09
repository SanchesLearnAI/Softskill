#!/bin/bash
#SBATCH --job-name=drop-i32-chunked-s1
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/drop_i32_chunked_seed1_%j.out
#SBATCH --error=logs/drop_i32_chunked_seed1_%j.err

set -uo pipefail

ROOT="${SOFTSKILL_WORKSPACE_ROOT:?Set SOFTSKILL_WORKSPACE_ROOT to the directory containing SoftSkill}"
PROJECT="$ROOT/SoftSkill"
PYTHON="$ROOT/envs/softskill/bin/python"
MODEL_SNAPSHOT="$ROOT/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATA_ROOT="$PROJECT/data/drop"
OUTPUT_DIR="$PROJECT/outputs/new_task_transfer/v1/drop/i32/seed1"
FINAL_CHECKPOINT="$OUTPUT_DIR/final.pt"
CHUNK_MODULE=skillopt.behavior_compression.new_task_transfer.v1.drop_chunked_eval
CHUNK_SIZE=512
MAX_MEAN_SAMPLE_SECONDS=2.0
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
  "$DATA_ROOT/raw/drop_dataset/drop_dataset_train.json" \
  "$DATA_ROOT/raw/drop_dataset/drop_dataset_dev.json" \
  "$PROJECT/skillopt/behavior_compression/new_task_transfer/v1/drop_chunked_eval.py" \
  "$OUTPUT_DIR/step0_audit.json" \
  "$OUTPUT_DIR/transfer_sanity.json" \
  "$OUTPUT_DIR/train_metrics.jsonl" \
  "$FINAL_CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "job=drop_i32_chunked_seed1 status=failed missing_required_file=$required" >&2
    exit 91
  fi
done

if [[ -s "$OUTPUT_DIR/summary.json" ]]; then
  if "$PYTHON" - "$OUTPUT_DIR/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "complete":
    raise SystemExit("existing DROP I32 summary is not complete")
PY
  then
    echo "job=drop_i32_chunked_seed1 status=already_complete"
    exit 0
  else
    exit 92
  fi
fi

TOTAL_SAMPLES="$($PYTHON - "$DATA_ROOT" <<'PY'
import sys
from skillopt.envs.drop.data import load_drop_split

print(len(load_drop_split(sys.argv[1], "validation")))
PY
)"
if [[ "$TOTAL_SAMPLES" != "9536" ]]; then
  echo "job=drop_i32_chunked_seed1 status=failed unexpected_validation_samples=$TOTAL_SAMPLES" >&2
  exit 93
fi

echo "job=drop_i32_chunked_seed1 event=start samples=$TOTAL_SAMPLES chunk_size=$CHUNK_SIZE time=$(date --iso-8601=seconds)"
chunk_index=0
for ((start=0; start<TOTAL_SAMPLES; start+=CHUNK_SIZE)); do
  end=$((start + CHUNK_SIZE))
  if ((end > TOTAL_SAMPLES)); then
    end=$TOTAL_SAMPLES
  fi
  phase_started="$(date --iso-8601=seconds)"
  echo "phase=chunk event=start index=$chunk_index start=$start end=$end time=$phase_started"
  if timeout --signal=TERM --kill-after=30s 30m \
    "$PYTHON" -m "$CHUNK_MODULE" worker \
    --model_name "$MODEL_SNAPSHOT" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --condition i32 \
    --seed 1 \
    --start "$start" \
    --chunk_size "$CHUNK_SIZE" \
    --max_prompt_tokens 8192 \
    --max_new_tokens 128; then
    exit_code=0
  else
    exit_code=$?
  fi
  echo "phase=chunk event=exit index=$chunk_index start=$start end=$end exit_code=$exit_code time=$(date --iso-8601=seconds)"
  if [[ $exit_code -ne 0 ]]; then
    echo "job=drop_i32_chunked_seed1 status=failed phase=chunk index=$chunk_index exit_code=$exit_code" >&2
    exit "$exit_code"
  fi

  if [[ $chunk_index -eq 0 ]]; then
    manifest="$OUTPUT_DIR/predictions/chunks/validation_$(printf '%06d' "$start")_$(printf '%06d' "$end").manifest.json"
    if ! "$PYTHON" - "$manifest" "$MAX_MEAN_SAMPLE_SECONDS" "$MAX_PEAK_RSS_KB" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_mean = float(sys.argv[2])
max_rss = int(sys.argv[3])
payload = json.loads(path.read_text(encoding="utf-8"))
progress = payload.get("progress", {})
recorded = int(progress.get("recorded_samples", 0))
expected = int(payload.get("sample_count", -1))
mean = progress.get("mean_sample_seconds")
peak_rss = progress.get("peak_vmrss_kb")
if recorded != expected or mean is None:
    raise SystemExit(f"first chunk has incomplete progress evidence: {recorded}/{expected}")
if float(mean) > max_mean:
    raise SystemExit(f"first chunk mean sample time {mean:.4f}s exceeds {max_mean:.4f}s")
if peak_rss is not None and int(peak_rss) > max_rss:
    raise SystemExit(f"first chunk peak RSS {peak_rss} KiB exceeds {max_rss} KiB")
print(f"first_chunk_gate=passed mean_sample_seconds={float(mean):.6f} peak_vmrss_kb={peak_rss}")
PY
    then
      echo "job=drop_i32_chunked_seed1 status=failed phase=first_chunk_gate" >&2
      exit 94
    fi
  fi
  chunk_index=$((chunk_index + 1))
done

echo "phase=merge event=start time=$(date --iso-8601=seconds)"
if timeout --signal=TERM --kill-after=30s 10m \
  "$PYTHON" -m "$CHUNK_MODULE" merge \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint "$FINAL_CHECKPOINT" \
  --condition i32 \
  --seed 1 \
  --chunk_size "$CHUNK_SIZE" \
  --max_prompt_tokens 8192 \
  --max_new_tokens 128; then
  echo "phase=merge event=complete time=$(date --iso-8601=seconds)"
else
  exit_code=$?
  echo "job=drop_i32_chunked_seed1 status=failed phase=merge exit_code=$exit_code" >&2
  exit "$exit_code"
fi

echo "phase=finalize_summary event=start time=$(date --iso-8601=seconds)"
if timeout --signal=TERM --kill-after=30s 15m \
  "$PYTHON" -m skillopt.behavior_compression.new_task_transfer.v1.train \
  --task drop \
  --condition i32 \
  --seed 1 \
  --output_dir "$OUTPUT_DIR" \
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
  --resume; then
  exit_code=0
else
  exit_code=$?
fi
echo "phase=finalize_summary event=exit exit_code=$exit_code time=$(date --iso-8601=seconds)"
if [[ $exit_code -ne 0 ]]; then
  exit "$exit_code"
fi

if "$PYTHON" - "$OUTPUT_DIR/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "complete" or int(payload.get("optimizer_steps", -1)) != 150:
    raise SystemExit("DROP I32 summary is missing or incomplete")
PY
then
  echo "job=drop_i32_chunked_seed1 status=complete time=$(date --iso-8601=seconds)"
else
  exit_code=$?
  echo "job=drop_i32_chunked_seed1 status=failed phase=summary_validation exit_code=$exit_code" >&2
  exit "$exit_code"
fi
