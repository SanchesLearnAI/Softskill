# SoftSkill Shared-Prefix Research Extension

> [!IMPORTANT]
> This is an undergraduate-stage exploratory reproduction and research-extension repository maintained by
> [@SanchesLearnAI](https://github.com/SanchesLearnAI). It is based on the original
> [SoftSkill repository](https://github.com/xijia-tao/SoftSkill) and paper by Xijia Tao
> et al. It is not the official SoftSkill repository and does not claim authorship of
> the original method, paper, upstream code, or upstream experimental results.

[![arXiv](https://img.shields.io/badge/arXiv-2606.20333-b31b1b.svg)](https://arxiv.org/abs/2606.20333)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

**Original SoftSkill method.** SoftSkill turns a natural-language skill file into a compact, trainable soft prefix for a frozen language model. The description and paper-level results below summarize the upstream work and are retained for reproduction context.

## Repository Provenance

This derivative repository contains four clearly separated provenance classes:

- **Upstream SoftSkill code:** files unchanged from upstream commit `4fc53008da110f354746bf36966dc0a2f44d3b92`.
- **Locally adapted upstream files:** upstream files with uncommitted local modifications, including extensions to soft-prefix modeling, training, configuration, documentation, and tests.
- **Local research additions:** files absent from the upstream commit, mainly shared/private prefix experiments, behavior-compression initialization, interventions, and ChartQA/DROP transfer experiments.
- **Third-party vendored or adapted code:** Microsoft SkillOpt-derived modules, SkillRL-derived ALFWorld wrappers, and SpreadsheetBench-adapted evaluation code.

See [PROVENANCE.md](PROVENANCE.md) for the interpretation rules,
[CODE_PROVENANCE.csv](CODE_PROVENANCE.csv) for the per-file classification, and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party sources. Git history
and filesystem state establish whether a file is upstream, modified, or newly added;
they do not by themselves prove the personal authorship of a local change.

## Undergraduate Exploratory Practice

This repository documents an exploratory practice carried out during undergraduate
study. Its purpose is to learn how soft-prefix methods can be reproduced, modified,
and evaluated. It is not an official implementation, a peer-reviewed contribution,
or a claim that the upstream SoftSkill method was created in this project.

The local exploration asks whether several tasks can share part of a fixed 32-token
prefix while retaining task-specific capacity:

- the experimental layout uses one trainable `Shared16` block and one `Task16` block
  per task, keeping the inference length at 32 tokens;
- the explored training schedule uses 30 shared-only warm-up optimizer steps followed
  by 120 joint optimizer steps with the base model frozen;
- for three tasks, the layout reduces the number of unique prefix vectors from
  `3 × 32 = 96` to `16 + 3 × 16 = 64`;
- additional exploratory code studies behavior-based initialization, prefix-state
  interventions, and transfer to ChartQA and DROP.

The current evidence does not isolate the effect of parameter sharing from behavior
initialization and the two-stage training schedule. Similar performance under the
combined setup therefore does not prove that sharing itself is effective, efficient,
or generally transferable. See [docs/undergraduate_exploration.md](docs/undergraduate_exploration.md)
for the presentation-derived project summary and limitations.

<p align="center">
  <img src="docs/assets/softskill-teaser.png" alt="SoftSkill method overview" width="95%">
</p>

SoftSkill keeps the backbone model frozen. It initializes a prefix from a readable skill document, trains only a small soft delta with next-token prediction over answers or successful trajectories, and selects deployed checkpoints by held-out task validation. The result is a reusable soft-skill artifact that can replace long prompt-side skill text at inference time.

## Why SoftSkill?

Modern agent systems increasingly rely on skill files, memories, or procedural instructions that tell a model how to inspect evidence, call tools, verify outputs, and avoid task-specific failure modes. These textual skills are portable and editable, but the model still has to translate them into behavior every time they are loaded.

SoftSkill asks a simple question: **can useful task behavior be internalized into a short continuous context while leaving the model weights untouched?** In the main QA setting, a length-32 SoftSkill prefix on `Qwen/Qwen3.5-4B` improves over no-skill prompting by 8.3 points on SearchQA, 42.1 points on LiveMath, and 1.3 points on DocVQA. Relative to SkillOpt text skills, SoftSkill improves accuracy by 5.2 points on SearchQA and 12.5 points on LiveMath while replacing long Markdown skill files with 32 virtual tokens.

<p align="center">
  <img src="docs/assets/qa-compression-diagnostics.png" alt="SoftSkill compression diagnostics for QA tasks" width="95%">
</p>

## Repository Overview

The upstream repository provides the public training and evaluation stack used for the SoftSkill release. This derivative retains that stack and adds local research extensions:

- SoftSkill prefix training and validation-selected checkpoint export.
- Prompt-embedding serving support for vLLM-style evaluation.
- Benchmark integrations for SearchQA, LiveMath, DocVQA, OfficeQA, SpreadsheetBench, and ALFWorld.
- Baseline and ablation configs for no-skill, hard-skill, LoRA, and soft-prefix runs.
- Compatibility entry points under the `skillopt` Python package.
- Released split manifests and scripts that materialize runnable benchmark data from upstream sources.

SoftSkill is strongest in the single-round QA setting in the paper. Agentic execution is included as a harder boundary case: trajectory imitation can provide useful signal, but long-horizon procedural behavior is still more fragile than answer-behavior compression.

## Repository Contents

```text
.
|-- configs/                 # Benchmark configs for baseline, LoRA, and SoftSkill runs
|-- data/
|   |-- README.md            # Released split manifests, source revisions, lookup keys
|   |-- *_id_split/          # Lightweight ID manifests for public benchmarks
|   `-- alfworld_path_split/  # ALFWorld path manifest
|-- scripts/
|   |-- data/                # Scripts that materialize runnable benchmark splits
|   |-- train/               # Public launch scripts for training and evaluation
|   |-- experiments/         # Research experiment launchers
|   `-- analysis/            # SoftSkill analysis utilities
|-- skillopt/                # Compatibility package and core training/evaluation code
|-- tests/                   # Unit and integration tests
|-- ckpt/                    # Checkpoint and skill notes; large artifacts excluded
`-- docs/                    # Additional project documentation
```

Generated rollouts, outputs, local corpora, model checkpoints, private environment files, and the separate `soft-skill/` paper repo are intentionally excluded from the public code release.

Machine-specific Slurm launchers require `SOFTSKILL_PROJECT_ROOT`,
`SOFTSKILL_WORKSPACE_ROOT`, and `SOFTSKILL_ACTIVATE`. Start from
`scripts/experiments/hpc_env.example.sh`; keep real paths outside version control.

## Install

Install the derivative repository in editable mode from a fresh checkout:

```bash
git clone https://github.com/SanchesLearnAI/Softskill.git
cd Softskill
pip install -e .
```

For SoftSkill training with Qwen models and local serving support, install the optional extras:

```bash
pip install -e ".[softprefix,qwen]"
```

For development and tests, include the development dependencies as well:

```bash
pip install -e ".[dev,softprefix]"
```

## Optional: Prompt-Embeds Serving

For larger evaluation runs, you can serve prompt embeddings through vLLM and point training or evaluation at that endpoint. This is optional for the default local-HF training command; use it when you set `INFERENCE_BACKEND=vllm_prompt_embeds`.

```bash
MODEL_NAME=Qwen/Qwen3.5-4B GPU_IDS=0 PORT=8010 bash scripts/train/start_server.sh
```

The server exposes an OpenAI-compatible endpoint at `http://127.0.0.1:8010/v1`.

To use that server with the shell launcher, pass its base URL explicitly:

```bash
INFERENCE_BACKEND=vllm_prompt_embeds \
INFERENCE_BASE_URL=http://127.0.0.1:8010 \
CONFIG=configs/searchqa/soft_prefix.yaml \
SPLIT_DIR=data/searchqa_split \
MODEL_NAME=Qwen/Qwen3.5-4B \
OUTPUT_DIR=outputs/SoftSkill_searchqa_example \
bash scripts/train/train_soft_prefix.sh
```

## Optional Credentials

The local SoftSkill training example below does **not** require OpenAI, Azure OpenAI, Anthropic, or MiniMax credentials. By default, `scripts/train/train_soft_prefix.sh` overrides the SearchQA config to use `INFERENCE_BACKEND=local_hf`, so training and evaluation run through the local Hugging Face model.

Configure credentials only when you use API-backed chat models, hard-skill reference evaluation, or trajectory rollouts that call an external backend:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Fill in only the backend variables you need. OpenAI-compatible endpoints reuse `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_AUTH_MODE=openai_compatible`.

## Prepare Benchmark Data

This repository includes lightweight split manifests under `data/*_id_split/` and `data/alfworld_path_split/`, but most benchmarks require materializing full examples from their original data sources before training or evaluation. Install the data helpers used by the preparation scripts:

The committed `*_id_split/` directories are lookup manifests; the `prepare_*` scripts resolve those IDs against upstream datasets and write the runnable `split_dir` directories used by training and evaluation.

```bash
pip install datasets huggingface_hub pillow
```

Prepare the text and math QA benchmarks:

```bash
python scripts/data/prepare_searchqa.py
python scripts/data/prepare_livemath.py
```

Prepare the remaining benchmark payloads used by the released configs:

```bash
python scripts/data/prepare_docvqa.py
python scripts/data/prepare_officeqa.py
python scripts/data/prepare_spreadsheetbench.py
```

`prepare_docvqa.py` writes `data/docvqa/splits` and saves images to `data/docvqa_images`. `prepare_officeqa.py` requires authorized Hugging Face access to the gated `databricks/officeqa` dataset; it writes `data/officeqa_split` and referenced Treasury Bulletin files under `data/officeqa_docs_official`. `prepare_spreadsheetbench.py` downloads and extracts SpreadsheetBench Verified 400 into `data/spreadsheetbench_verified_400` and writes `data/spreadsheetbench_split`.

ALFWorld is different: the released split is already a path manifest at `data/alfworld_path_split`, but the game files must be downloaded separately. Install the ALFWorld extra, download the games, set `ALFWORLD_DATA`, then validate that the manifest resolves:

```bash
pip install -e ".[alfworld]"
alfworld-download
export ALFWORLD_DATA=/path/to/alfworld/data
python scripts/data/prepare_alfworld.py --data_root "$ALFWORLD_DATA"
```

See `data/README.md` for split counts, source revisions, and the lookup keys used by each manifest.

## Train a SoftSkill

The SearchQA example trains two length-32 soft skills together for `Qwen/Qwen3.5-4B` using the materialized split directory in `data/searchqa_split`. Both are initialized from consecutive token embeddings of the same Markdown skill, concatenated in the model context, and updated by the same next-token loss. The backbone stays frozen; the output directory receives the learned skill checkpoints and training artifacts.

```bash
CONFIG=configs/searchqa/soft_prefix.yaml \
SPLIT_DIR=data/searchqa_split \
MODEL_NAME=Qwen/Qwen3.5-4B \
OUTPUT_DIR=outputs/SoftSkill_searchqa_example \
bash scripts/train/train_soft_prefix.sh
```

You can also call the Python entry point directly:

```bash
python scripts/train_soft_prefix.py \
  --config configs/searchqa/soft_prefix.yaml \
  --split_dir data/searchqa_split \
  --model_name Qwen/Qwen3.5-4B \
  --out_root outputs/SoftSkill_searchqa_example
```

## Evaluate a SoftSkill

After training, evaluate the best checkpoint on the configured validation split:

```bash
CHECKPOINT_PATH=outputs/SoftSkill_searchqa_example/best_prefix.pt \
OUTPUT_DIR=outputs/SoftSkill_searchqa_eval \
bash scripts/train/eval_soft_prefix.sh
```

Hard-skill reference evaluation remains available through:

```bash
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill ckpt/searchqa/gpt5.5_skill.md \
  --split valid_unseen \
  --split_dir data/searchqa_split
```

## License and Attribution

The upstream code is distributed under the MIT License, whose Microsoft copyright and permission notice are preserved verbatim in [LICENSE](LICENSE). This derivative also preserves attribution to the original SoftSkill paper/repository and Microsoft [SkillOpt](https://github.com/microsoft/SkillOpt/). Some subtrees contain additional third-party adaptations; consult [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution.

Local modifications are offered under the same MIT terms only to the extent that the
person publishing them has the right to do so. Repository metadata cannot establish
employment, laboratory, sponsorship, or coauthor ownership; confirm those rights
before making the repository public.

## Citation

If you find SoftSkill useful, please cite:

```bibtex
@misc{tao2026softskillbehavioralcompressioncontextual,
      title={SoftSkill: Behavioral Compression for Contextual Adaptation},
      author={Xijia Tao and Yihua Teng and Xinyu Fu and Ziru Liu and Kecheng Chen and Yuzhi Zhao and Suiyun Zhang and Rui Liu and Lingpeng Kong},
      year={2026},
      eprint={2606.20333},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.20333},
}
```
