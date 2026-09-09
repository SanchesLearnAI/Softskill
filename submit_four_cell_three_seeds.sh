#!/usr/bin/env bash
set -euo pipefail

cd "${SOFTSKILL_PROJECT_ROOT:?Set SOFTSKILL_PROJECT_ROOT to the SoftSkill checkout}"
mkdir -p logs

for family in A_direct B_residual C_direct_shared D_residual_shared; do
  /opt/slurm/bin/sbatch \
    --job-name="${family}-3seeds" \
    --export=ALL,FAMILY="${family}" \
    train_four_cell_three_seeds.slurm
done
