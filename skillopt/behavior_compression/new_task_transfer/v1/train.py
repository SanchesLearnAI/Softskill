"""Single-task final-only training and evaluation entry."""
from __future__ import annotations

import argparse
import json
import random

from skillopt.behavior_compression.new_task_transfer.v1.model import (
    TargetOnlyTransferSoftPrefixVisionLM,
)
from skillopt.behavior_compression.new_task_transfer.v1.final_protocol import (
    run_final_only_experiment,
    run_training_to_final_checkpoint,
)
from skillopt.behavior_compression.new_task_transfer.v1.train_core import (
    MinimalTrainConfig,
)
from skillopt.envs.chartqa.data import (
    ChartQAPrefixDataset,
    select_chartqa_transfer_splits,
)
from skillopt.envs.drop.data import DropPrefixDataset, select_drop_transfer_splits
from skillopt.softprefix.data import PrefixBatchCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("chartqa", "drop"))
    parser.add_argument("--condition", required=True, choices=("f0", "f1", "i32"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--source_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_optimizer_steps", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--train_size", type=int, default=0)
    parser.add_argument("--max_prompt_tokens", type=int, default=8192)
    parser.add_argument("--max_target_tokens", type=int, default=128)
    parser.add_argument("--max_image_tokens", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--checkpoint_every_steps", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_max_new_tokens", type=int, default=0)
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="stop after validating the step-150 final checkpoint; do not evaluate",
    )
    return parser.parse_args()


def _build_real_training_components(args: argparse.Namespace):
    model = TargetOnlyTransferSoftPrefixVisionLM(
        args.model_name,
        task_name=args.task,
        condition=args.condition,
        source_checkpoint_path=args.source_checkpoint or None,
        torch_dtype="auto",
        device="auto",
        trust_remote_code=True,
    )
    train_size = int(args.train_size) or None
    if args.task == "chartqa":
        root = args.data_root or "data/chartqa"
        splits, _manifest = select_chartqa_transfer_splits(
            root,
            train_size=train_size,
            seed=args.seed,
        )
        dataset = ChartQAPrefixDataset(
            splits["train"],
            model.processor,
            model.tokenizer,
            max_prompt_tokens=args.max_prompt_tokens,
            max_target_tokens=args.max_target_tokens,
            max_image_tokens=args.max_image_tokens,
        )
    else:
        root = args.data_root or "data/drop"
        splits, _manifest = select_drop_transfer_splits(
            root,
            train_size=train_size,
            seed=args.seed,
        )
        dataset = DropPrefixDataset(
            splits["train"],
            model.tokenizer,
            max_prompt_tokens=args.max_prompt_tokens,
            max_target_tokens=args.max_target_tokens,
        )
    collator = PrefixBatchCollator(model.tokenizer.pad_token_id)
    return model, dataset, collator, splits


def main() -> None:
    args = parse_args()
    config = MinimalTrainConfig(
        task=args.task,
        condition=args.condition,
        seed=args.seed,
        source_checkpoint=args.source_checkpoint,
        output_dir=args.output_dir,
        max_optimizer_steps=args.max_optimizer_steps,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        batch_size=args.batch_size,
    ).validated()
    random.seed(config.seed)
    model, dataset, collator, splits = _build_real_training_components(args)
    model.torch.manual_seed(config.seed)
    generation_config = {
        "max_prompt_tokens": args.max_prompt_tokens,
        "temperature": args.eval_temperature,
        "max_image_tokens": args.max_image_tokens,
        "enable_thinking": False,
    }
    if args.eval_max_new_tokens:
        generation_config["max_new_tokens"] = args.eval_max_new_tokens
    if args.train_only:
        result = run_training_to_final_checkpoint(
            config=config,
            model=model,
            train_dataset=dataset,
            collator=collator,
            resume=args.resume,
            resume_checkpoint=args.resume_checkpoint,
            checkpoint_every_steps=args.checkpoint_every_steps,
        )
    else:
        result = run_final_only_experiment(
            config=config,
            model=model,
            train_dataset=dataset,
            collator=collator,
            evaluation_splits=splits,
            eval_batch_size=args.eval_batch_size,
            generation_config=generation_config,
            resume=args.resume,
            resume_checkpoint=args.resume_checkpoint,
            checkpoint_every_steps=args.checkpoint_every_steps,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
