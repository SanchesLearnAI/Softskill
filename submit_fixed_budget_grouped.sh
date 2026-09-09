#!/usr/bin/env bash
set -euo pipefail

cd "${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
mkdir -p logs

for task in searchqa livemath docvqa; do
  /opt/slurm/bin/sbatch \
    --job-name="2x16-${task}-3seeds" \
    --export=ALL,FAMILY=two_by_16,TASK="${task}" \
    train_fixed_budget_three_seeds.slurm
done

/opt/slurm/bin/sbatch \
  --job-name="shared16-task16-3seeds" \
  --export=ALL,FAMILY=shared16_task16 \
  train_fixed_budget_three_seeds.slurm
