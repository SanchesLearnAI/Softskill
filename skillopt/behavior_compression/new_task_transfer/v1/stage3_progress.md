# Stage 3 progress

Last recorded: 2026-08-17

## ChartQA chunked recovery

- Added `chartqa_chunked_eval.py` with condition/split-aware process isolation, resumable 256-sample chunks, strict merge, and two-split evaluation-ledger publication.
- Added `train.py --train_only`, backed by `run_training_to_final_checkpoint`, so F1 and I32 can produce and validate a step-150 `final.pt` without entering monolithic evaluation.
- Updated `train_new_task_transfer_chartqa_seed1.slurm` for serial F0/F1/I32 execution, per-condition training, chunked validation/test, resource gates, merge, and final-only summary publication.
- Added `tests/test_chartqa_chunked_eval_v1.py` and a training-only protocol test. The remote record reports 37/37 directed regressions and then 13/13 final ChartQA/protocol tests passing.
- Real-data preflight reported 7,398/960/1,250 human train/validation/test samples. F0 step 150 and validation were reusable; F1/I32 output directories were absent; the source Shared tensor shape was `[1, 16, 2560]`.
- Submitted A800 job `10245176` on 2026-08-17. Its last recorded state was `PENDING (Priority)`.

## DROP I32 chunked recovery

- Added `drop_chunked_eval.py` with deterministic 512-sample process-isolated workers, resumable chunks, strict checkpoint/sample/hash validation, official-metric revalidation, atomic merge, and evaluation-ledger publication.
- Added `tests/test_drop_chunked_eval_v1.py`. The remote record reports 6/6 chunk-specific tests and 32/32 combined directed regressions passing.
- Real-data preflight found a duplicate official DROP validation query ID at positions 307 and 320. Progress identity therefore uses chunk-local sample index plus positional ID so both records remain distinct.
- Updated `resume_drop_i32_seed1.slurm` for 19 sequential worker processes, bounded per-chunk timeout, a first-chunk throughput/RSS gate, merge, and final-only summary publication.
- Real-data preflight reported checkpoint SHA256 `ef3d448b452c031f096700cbb58992385495a7ecc7a9a4216d3c86828ab29d0b`, 9,536 ordered validation records, 19 chunks, a frozen base model, prefix length 32, and residual mode disabled.
- Submitted A800 job `10244994` on 2026-08-17. Its last recorded state was `PENDING (Priority)`.

## Completed implementation

- `train.py`: minimal ChartQA/DROP CLI plus explicit training-only mode.
- `train_core.py`: constant-learning-rate AdamW loop, answer-token-weighted accumulation, exact optimizer-step control, deterministic task/seed batch schedules, and sample-order hashes.
- `train_state.py`: step-0 audit, fail-fast startup checks, first-step gradient/update sanity checks, latest/periodic/final checkpoints, and explicit resume support.
- `evaluation.py`: separate ChartQA/DROP generation, official metric reuse, ordered predictions, and atomic failure cleanup.
- `final_protocol.py`: strict step-150 final-only evaluation, metrics/predictions/summary closure, and resume deduplication.
- Directed tests cover the training loop, state/resume behavior, evaluation, final protocol, and chunked recovery paths.

## Pending work

- Confirm the final scheduler outcome of jobs `10245176` and `10244994`; the status above is only the 2026-08-17 submission snapshot.
- Recover or rerun ChartQA GPU evaluation and publish final F0/F1/I32 metrics.
- Recover or rerun DROP I32 GPU evaluation and publish final metrics.
- Run the full local test suite in an environment with the development dependencies installed. The 2026-09-08 handoff machine completed Python bytecode compilation, but its bundled Python environment did not include `pytest`.
