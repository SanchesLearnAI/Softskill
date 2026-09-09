#!/usr/bin/env bash
set -euo pipefail

cd "${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"

for task in searchqa livemath docvqa; do
  for seed in 1 2 3; do
    /opt/slurm/bin/sbatch \
      --job-name="2x16-${task}-s${seed}" \
      --export=ALL,FAMILY=two_by_16,TASK="${task}",SEED="${seed}" \
      train_fixed_budget_single.slurm
  done
done

for seed in 1 2 3; do
  /opt/slurm/bin/sbatch \
    --job-name="shared16-task16-s${seed}" \
    --export=ALL,FAMILY=shared16_task16,SEED="${seed}" \
    train_fixed_budget_single.slurm
done
