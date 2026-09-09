# Local Change Summary

## Classification snapshot

The original review, before adding release-preparation documentation, found:

- 276 paths tracked by upstream commit `4fc53008da110f354746bf36966dc0a2f44d3b92`;
- 17 upstream files modified in the local working tree;
- 95 paths added locally and absent from that upstream commit.

Release preparation then added provenance documents, an HPC environment example, and
repository metadata changes. The current exact per-file state is recorded in
`CODE_PROVENANCE.csv`; regenerate that table after future edits.

## Main locally adapted upstream areas

- `skillopt/softprefix/model.py`: substantial model changes for prefix initialization,
  composition, and training behavior.
- `skillopt/softprefix/data.py`, `trainer.py`, `transfer.py`, and
  `vllm_prompt_embeds.py`: supporting data, training, transfer, and serving changes.
- `scripts/train_soft_prefix.py`: training entry-point changes.
- `configs/*/soft_prefix.yaml`: local experiment configuration changes.
- `tests/test_docvqa_softprefix_data.py` and `tests/test_softprefix_model_loss.py`:
  regression coverage for the modified upstream behavior.
- `README.md`, `CONTRIBUTING.md`, `mkdocs.yml`, and `pyproject.toml`: derivative-repository
  identity, attribution, and publication metadata.

## Main local additions

- `skillopt/softprefix/multitask_v2.py`, `multitask_trainer.py`,
  `interventions_v2.py`, and `new_task_transfer.py`;
- `skillopt/behavior_compression/` including behavior initialization, audits, and the
  ChartQA/DROP transfer protocol;
- `skillopt/envs/chartqa/` and `skillopt/envs/drop/`;
- joint-training, fixed-budget, hybrid, intervention, analysis, and recovery scripts;
- corresponding configs, documentation, and tests.

These classifications identify technical derivation. They intentionally do not state
that the GitHub account owner personally authored every local change. Replace this
caveat with an accurate contributor list only after confirming who did the work.
