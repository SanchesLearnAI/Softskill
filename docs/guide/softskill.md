# Soft-Prefix Workflow

SoftSkill keeps the SkillOpt-compatible import path `skillopt` while adding a
soft-prefix training stack under `skillopt/softprefix/`.

## Install

```bash
pip install -e ".[dev,softprefix]"
```

Install `.[qwen]` as well when using local vLLM/Qwen serving.

## Serve Prompt Embeddings

```bash
MODEL_NAME=Qwen/Qwen3.5-4B GPU_IDS=0 PORT=8010 bash scripts/train/start_server.sh
```

The server exposes an OpenAI-compatible endpoint at
`http://127.0.0.1:8010/v1`.

## Train

The released SoftSkill configs train two soft skills together. Each skill has
`prefix_length` rows and both are initialized from consecutive token
embeddings of the same Markdown skill document. They are concatenated for one
forward pass, so the same next-token loss updates all three parameter blocks:

```yaml
soft_prefix:
  num_soft_skills: 2
  prefix_length: 32
  init_strategy: text
```

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

## Evaluate

```bash
CHECKPOINT_PATH=outputs/SoftSkill_searchqa_example/best_prefix.pt \
OUTPUT_DIR=outputs/SoftSkill_searchqa_eval \
bash scripts/train/eval_soft_prefix.sh
```

Use `scripts/eval_only.py` with `ckpt/*/gpt5.5_skill.md` when comparing
against retained hard-skill SkillOpt reference artifacts.

## Joint Shared + Task Training

The first multi-task experiment uses one shared length-32 prefix and one
length-32 Markdown-initialized prefix for the active task. SearchQA, LiveMath,
and DocVQA batches are sampled in balanced round-robin cycles. The shared
learning rate decays exponentially while the task learning rate rises smoothly:

```bash
python scripts/train_shared_task_soft_prefix.py \
  --out_root outputs/shared_task/seed1 \
  --seed 1 \
  --prefix_length 32
```

Each example therefore uses 64 virtual tokens. The frozen vision-capable Qwen
backbone handles both text-only batches and DocVQA image batches, so the shared
prefix is genuinely the same parameter across all three tasks.
