# Corrected Joint SoftSkill v2 Protocol

The v2 implementation is isolated from the legacy joint trainer so historical
and in-flight v1 results remain reproducible.

## Four-cell comparison

| Cell | Prefix layout | Reparameterization |
| --- | --- | --- |
| A | Two task-specific 16-token blocks | Direct |
| B | Two task-specific 16-token blocks | One matched residual MLP |
| C | Shared 16-token block + task-specific 16-token block | Direct |
| D | Shared 16-token block + task-specific 16-token block | One matched residual MLP |

Every cell exposes exactly 32 virtual tokens at inference. Residual cells use
one common bottleneck MLP over the complete 32-token prompt, so B and D have
the same reparameterizer parameter count. The MLP is folded into the prompt
artifact after training and is not required for inference.

This matched global MLP is a controlled training-time component shared by B
and D. Consequently, B has independent prompt embeddings but is not a claim
that absolutely no trainable parameter is shared across tasks.

## Initialization

For task `t`, its Markdown skill is tokenized once and split into two
consecutive embedding blocks, `E_t[0:16]` and `E_t[16:32]`.

- A/B use both task-specific blocks.
- C/D initialize the shared block with the order-invariant, position-wise mean
  of every task's first block. Each position is norm-calibrated to the mean norm
  of its contributors. The private task block uses that task's second block.

This prevents the first task in the task list from dominating shared-prefix
initialization and avoids duplicating one Markdown block in both prefix slots.

### Behavior-Markdown initialization

The optional `behavior_markdown` mode replaces positional averaging with an
explicit semantic factorization:

```text
skillopt/behavior_compression/v1/shared_behavior.md   -> Shared16
skillopt/behavior_compression/v1/searchqa_behavior.md -> SearchQA Task16
skillopt/behavior_compression/v1/livemath_behavior.md -> LiveMath Task16
skillopt/behavior_compression/v1/docvqa_behavior.md   -> DocVQA Task16
```

The independent layout stores `[Shared16, Task16]` separately for every task.
The shared layout stores one Shared16 plus one Task16 per task. Both layouts
therefore start from exactly the same behavior information; their difference
is whether the common behavior is duplicated or genuinely shared.

The seed files start with behavior-bearing text rather than headings or
placeholders. Each stripped file is exactly 16 Qwen/Qwen3.5-4B tokens.
`tokenizer_audit.json` and every run's `protocol_audit.json` record the raw
text, complete and selected token ids, decoded selection, and whether the
initializer repeated or truncated the source. Checkpoints also embed the
initialization text so the four-state exporter can reconstruct the exact
starting point after source files change.

The three source `ckpt/*/gpt5.5_skill.md` files were added as artifacts without
their generation logs. The current SkillOpt generator selects candidate skills
on `valid_seen`, so train-only provenance cannot be certified. The exact
evidence and runtime source hashes are preserved in `provenance.json` and each
protocol audit. Behavior v1 is therefore an exploratory, validation-influenced
result rather than a final leakage-free claim.

## Joint optimizer step

The task order is fixed as SearchQA, LiveMath, then DocVQA. Before each single
optimizer step, v2 consumes the configured number of microbatches:

| Task | Batch size | Accumulation | Effective batch |
| --- | ---: | ---: | ---: |
| SearchQA | 8 | 1 | 8 |
| LiveMath | 4 | 2 | 8 |
| DocVQA | 1 | 4 | 4 |

Microbatch losses are weighted by their actual number of supervised target
tokens within each task, then the three task losses are macro-averaged. Thus
every task contributes one third of a joint update without overweighting a
short final batch.

The strict four-cell comparison defaults to a uniform learning rate. The
progressive shared-decay/task-growth schedule remains available as a separate
Shared+Task ablation through `LR_SCHEDULE=progressive`; it is not mixed into
the main A/B/C/D comparison. Progressive residual runs automatically use
`branch_decoupled`: one shared residual MLP follows the shared schedule and one
MLP per task follows the task schedule. At progress zero, the complete task
path is frozen.

Behavior compression v1 uses a separate deterministic `behavior_two_stage`
schedule with residual reparameterization forbidden. For the first 30 of 150
optimizer steps only Shared16 is trainable at `1e-3`. For the remaining 120
steps all Task16 blocks are trainable, Shared16 uses `1e-4`, and Task16 uses
`1e-3`. Stage transitions, `requires_grad` state, and both optimizer-group
learning rates are persisted in `training_stage_events.json` and
`protocol_audit.json`.

Residual MLPs are initialized from an isolated deterministic RNG stream. This
makes B and D use identical global-MLP initialization for the same experiment
seed even though their prompt tensor layouts differ.

## Running one cell

```bash
FAMILY=D_residual_shared SEED=1 \
  bash scripts/experiments/train_v2_four_cell.sh
```

Outputs are written below `outputs/v2/four_cell/` and never overwrite v1
artifacts. Progressive ablations are written below `outputs/v2/progressive/`.
Every run writes `protocol_audit.json` containing resolved configs, Markdown
and initialization hashes, model revision, parameter counts, accumulation,
and package versions. The Slurm wrapper `train_v2_four_cell.slurm` accepts
`FAMILY` and an optional space-separated `SEEDS` value, but it is not submitted
automatically.

Run the first behavior-compression v1 experiment with:

```bash
bash scripts/experiments/train_behavior_compression_v1.sh
```

The output is isolated below
`outputs/behavior_compression/v1/shared16_task16_raw_two_stage/seed1/`; old v2
paths are not overwritten. The launcher requires the resolved loaders to
produce exactly 150 optimizer steps.

## Four-state prefix attribution

New v2 runs also save `initial_prefix_states.pt` before the first optimizer
step and `prefix_states.pt` after restoring the best validation checkpoint.
The latter contains, for every task:

- `initial`: raw Markdown-initialized prompt.
- `final_raw`: trained prompt before residual reparameterization.
- `residual_contribution`: `folded - final_raw`.
- `folded`: the exact prompt used for inference.

This makes `final_raw - initial` attributable to joint training and
`folded - final_raw` attributable to the residual MLP. Reconstruction and
shared-block equality errors are stored as invariants.

For an older v2 checkpoint, backfill the artifact without training or dataset
loading (the frozen model is loaded only to reproduce Markdown embeddings):

```bash
python scripts/export_prefix_states_v2.py \
  --checkpoint outputs/v2/four_cell/B_joint_2x16_residual/seed1/best_v2.pt
```

After both B and D have `prefix_states.pt`, run the small CPU-only analysis:

```bash
python scripts/analyze_prefix_state_trajectory.py \
  --independent-states outputs/v2/four_cell/B_joint_2x16_residual/seed1/prefix_states.pt \
  --shared-states outputs/v2/four_cell/D_shared16_task16_residual/seed1/prefix_states.pt \
  --output-dir outputs/v2/prefix_state_trajectory/B_vs_D_seed1
```

The analysis writes decomposition, transition, structural pairwise, aggregate
CSV files, plus one complete JSON artifact. The parameterized CPU wrapper is
`analyze_prefix_state_trajectory_cpu.slurm`.
