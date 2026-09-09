# SoftSkill Shared-Prefix Research Extension

> This documentation belongs to an undergraduate-stage exploratory reproduction
> and research extension maintained at `SanchesLearnAI/Softskill`. The original SoftSkill
> method, paper, and upstream code are credited in the repository-level
> `PROVENANCE.md` and `AUTHORS.md` files.

SoftSkill is a research codebase for training soft-prefix skills for frozen
language and vision-language models.

The upstream project is SoftSkill. The Python import package remains `skillopt`
for compatibility with the SkillOpt-derived codebase, configs, scripts, and
checkpoint skills.

The local work studies a `Shared16 + Task16` layout under a fixed 32-token
inference budget. It should be read as exploratory practice rather than a verified
claim that shared prefixes are causally effective or generally transferable.

## Start Here

- Install with `pip install -e ".[dev,softprefix]"`.
- Train soft prefixes with `scripts/train_soft_prefix.py` or the scripts in
  `scripts/train/`.
- Evaluate retained hard-skill baselines with `scripts/eval_only.py` and the
  reference skills in `ckpt/`.
