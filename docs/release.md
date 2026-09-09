# Release Boundary

This derivative repository is intended to release an independently maintained
SoftSkill/SoftPrefix reproduction and research extension,
lightweight split manifests, tests, and reference SkillOpt-compatible skill
artifacts needed for reproduction.

## Included

- `skillopt/`: compatibility import package containing the training stack,
  environment adapters, and SoftSkill soft-prefix implementation.
- `skillopt/softprefix/`: soft-prefix, LoRA, transfer, and prompt-embedding
  serving code.
- `configs/`: public configs for hard-skill, soft-prefix, and LoRA runs.
- `scripts/train/`, `scripts/experiments/`, and `scripts/analysis/`: public
  launch and analysis helpers.
- `ckpt/`: retained SkillOpt reference skill markdown files for baseline
  evaluation and comparison.
- `data/*_split/` and `data/*_path_split/`: lightweight split manifests only.

## Excluded

- `.env`, `.secrets/`, and local credential material.
- `outputs/`, `rollouts/`, logs, caches, and package build artifacts.
- Full benchmark corpora, local dataset checkouts, and private payloads.
- The separate `soft-skill/` paper repository.

## Attribution

The original SoftSkill method, paper, authors, and upstream repository are
identified in `PROVENANCE.md` and `AUTHORS.md`. SoftSkill is derived from
Microsoft SkillOpt under the MIT License. Retained SkillOpt and other
third-party adaptations are called out in `THIRD_PARTY_NOTICES.md` and local
source notices where they remain part of the release.
