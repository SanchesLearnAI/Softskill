#!/usr/bin/env bash
#SBATCH --job-name=prefix-state-pipeline
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prefix_state_pipeline_%j.out
#SBATCH --error=logs/prefix_state_pipeline_%j.err

set -euo pipefail

PROJECT_ROOT="${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
cd "${PROJECT_ROOT}"
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
INDEPENDENT_ROOT="${INDEPENDENT_ROOT:-outputs/v2/four_cell/B_joint_2x16_residual/seed1}"
SHARED_ROOT="${SHARED_ROOT:-outputs/v2/four_cell/D_shared16_task16_residual/seed1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/v2/prefix_state_trajectory/B_vs_D_seed1}"

for run_root in "${INDEPENDENT_ROOT}" "${SHARED_ROOT}"; do
  checkpoint="${run_root}/best_v2.pt"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  python scripts/export_prefix_states_v2.py \
    --checkpoint "${checkpoint}" \
    --model-name "${MODEL_NAME}" \
    --device cpu \
    --overwrite
done

python scripts/analyze_prefix_state_trajectory.py \
  --independent-states "${INDEPENDENT_ROOT}/prefix_states.pt" \
  --shared-states "${SHARED_ROOT}/prefix_states.pt" \
  --output-dir "${OUTPUT_DIR}"

test -s "${OUTPUT_DIR}/prefix_state_analysis.json"
test -s "${OUTPUT_DIR}/prefix_state_decomposition.csv"
test -s "${OUTPUT_DIR}/prefix_state_structure.csv"

echo "PREFIX_STATE_PIPELINE_OK output=${OUTPUT_DIR}"
