# Third-Party Notices

This repository preserves and extends a codebase with multiple upstream sources.

## Microsoft SkillOpt

- Source: <https://github.com/microsoft/SkillOpt>
- Role: base optimizer, benchmark, model-backend, prompt, and compatibility code
- License represented in this repository: MIT
- Required action: retain the root `LICENSE` notice in copies and substantial portions

## Original SoftSkill repository and paper

- Source code: <https://github.com/xijia-tao/SoftSkill>
- Paper: <https://arxiv.org/abs/2606.20333>
- Role: soft-prefix method, training/evaluation stack, documentation, configs, and released results
- Required action: retain repository and paper attribution; do not present the method or reported paper results as original work of this derivative repository

## SkillRL-derived ALFWorld wrapper

- Path: `skillopt/envs/alfworld/vendor/`
- Source recorded in the code: `https://github.com/NTU-LANTERN/SkillRL`
- License recorded in the code: Apache-2.0
- Local modification recorded in the code: imports use the pip-installed ALFWorld package
- Required action before public release: verify the upstream URL/revision and include any license/NOTICE text required by the exact copied revision

## SpreadsheetBench adaptations

- Paths: `skillopt/envs/spreadsheetbench/evaluator.py` and `react_agent.py`
- Source recorded in the code: <https://github.com/RUCKBReasoning/SpreadsheetBench>
- Required action before public release: verify the exact source revision and applicable license for the adapted portions

## Benchmark data

The repository contains lightweight identifiers and manifests, not a blanket license to
redistribute complete benchmark datasets. Follow the original dataset terms listed in
`data/README.md` and the corresponding preparation scripts.
