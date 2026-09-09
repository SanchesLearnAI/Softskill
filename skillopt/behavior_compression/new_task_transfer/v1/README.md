# New-task transfer behavior initializations

These one-line texts initialize the task-specific 16-token branch for the
ChartQA and DROP transfer experiments.  They were tokenized with
`Qwen/Qwen3.5-4B`, `add_special_tokens=False`; the exact IDs are recorded in
`tokenizer_audit.json`.  They describe task behavior only and were written
without using validation or test feedback.

## Stage 2 prefix structures

`model.py` implements target-only F0, F1, and I32 by subclassing the existing
`SoftPrefixVisionLM`, so its text/vision prefix injection path is reused.  The
base model is frozen, residual reparameterization is disabled, and every
condition exposes exactly 32 virtual tokens.

`checkpoint.py` accepts only the explicit key
`checkpoint["model"]["shared_prefix_embeddings"]`.  The existing v2 files
store it as `[1, 16, 2560]`; extraction removes only that singleton outer
dimension and then requires `[16, hidden_size]` plus an exact dtype match.
Task prefixes, optimizer state, scheduler state, and history are never loaded.

Verified source candidates (documentation only; F1 never auto-selects one):

- `outputs/behavior_compression/v1/shared16_task16_raw_two_stage/seed1/best_v2.pt`
- `outputs/behavior_compression/v1/shared16_task16_raw_two_stage/seed2/best_v2.pt`
- `outputs/behavior_compression/v1/shared16_task16_raw_two_stage/seed3/best_v2.pt`
