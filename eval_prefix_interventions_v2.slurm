#!/usr/bin/env bash
#SBATCH --job-name=v2-D-intervene
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v2_D_interventions_%j.out
#SBATCH --error=logs/v2_D_interventions_%j.err

set -euo pipefail

PROJECT_ROOT="${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
cd "${PROJECT_ROOT}"
source "${SOFTSKILL_ACTIVATE:?Set SOFTSKILL_ACTIVATE to the environment activation script}"
mkdir -p logs outputs/v2/interventions

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE="${STAGE:-validation}"
CHECKPOINT="${CHECKPOINT:-outputs/v2/four_cell/D_shared16_task16_residual/seed1/best_v2.pt}"
PREFIX_STATES="${PREFIX_STATES:-outputs/v2/four_cell/D_shared16_task16_residual/seed1/prefix_states.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/v2/interventions/D_shared16_task16_residual/seed1}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-${OUT_ROOT}/selection_manifest.json}"

test -s "${CHECKPOINT}"
test -s "${PREFIX_STATES}"

args=(
  --stage "${STAGE}"
  --checkpoint "${CHECKPOINT}"
  --prefix-states "${PREFIX_STATES}"
  --out-root "${OUT_ROOT}"
  --model-name "${MODEL_NAME:-Qwen/Qwen3.5-4B}"
  --seed "${SEED:-1}"
  --alphas 0 0.015625 0.03125 0.0625 0.125 0.25 0.5 1
)

if [[ "${STAGE}" == "test" ]]; then
  test -s "${SELECTION_MANIFEST}"
  args+=(--selection-manifest "${SELECTION_MANIFEST}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  args+=(--resume)
fi

echo "Starting v2-D interventions stage=${STAGE} at $(date --iso-8601=seconds)"
python scripts/eval_prefix_interventions_v2.py "${args[@]}"
test -s "${OUT_ROOT}/${STAGE}/completed.json"
test -s "${OUT_ROOT}/${STAGE}/intervention_summary.csv"
if [[ "${STAGE}" == "validation" ]]; then
  test -s "${SELECTION_MANIFEST}"
fi
echo "V2_D_INTERVENTIONS_OK stage=${STAGE} output=${OUT_ROOT}"
