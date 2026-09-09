"""CLI for evaluating native and crossed F0/F1 Shared16+Task16 prefixes."""
from __future__ import annotations

import argparse
import json

import torch

from skillopt.behavior_compression.new_task_transfer.v1.cross_composition import (
    DEFAULT_COMPOSITIONS,
    COMPOSITION_SOURCES,
    deterministic_subset,
    load_frozen_transfer_checkpoint,
    run_cross_composition_evaluation,
    validate_checkpoint_pair,
)
from skillopt.behavior_compression.new_task_transfer.v1.model import (
    TargetOnlyTransferSoftPrefixVisionLM,
)
from skillopt.envs.chartqa.data import select_chartqa_transfer_splits
from skillopt.envs.drop.data import select_drop_transfer_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("chartqa", "drop"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--f0_checkpoint", required=True)
    parser.add_argument("--f1_checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_prompt_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_image_tokens", type=int, default=0)
    parser.add_argument(
        "--compositions",
        nargs="+",
        choices=tuple(COMPOSITION_SOURCES),
        default=list(DEFAULT_COMPOSITIONS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    f0 = load_frozen_transfer_checkpoint(
        args.f0_checkpoint,
        expected_condition="f0",
        expected_task=args.task,
        expected_seed=args.seed,
        torch_module=torch,
    )
    f1 = load_frozen_transfer_checkpoint(
        args.f1_checkpoint,
        expected_condition="f1",
        expected_task=args.task,
        expected_seed=args.seed,
        torch_module=torch,
    )
    validate_checkpoint_pair(f0, f1)
    model_name = str(args.model_name).strip() or f0.model_name
    if model_name != f0.model_name:
        raise RuntimeError(
            "--model_name must exactly match the model recorded in both checkpoints"
        )
    split = str(args.split).strip().lower() or (
        "test" if args.task == "chartqa" else "validation"
    )
    if args.task == "chartqa":
        splits, _manifest = select_chartqa_transfer_splits(
            args.data_root, seed=args.seed
        )
    else:
        splits, _manifest = select_drop_transfer_splits(
            args.data_root, seed=args.seed
        )
    if split not in splits:
        raise ValueError(f"split {split!r} is unavailable for {args.task}")
    dataset, dataset_id_hash = deterministic_subset(
        splits[split],
        task=args.task,
        split=split,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    model = TargetOnlyTransferSoftPrefixVisionLM(
        model_name,
        task_name=args.task,
        condition="f0",
        source_checkpoint_path=None,
        torch_dtype="auto",
        device="auto",
        trust_remote_code=True,
    )
    generation_config = {
        "max_prompt_tokens": args.max_prompt_tokens,
        "temperature": args.temperature,
        "max_image_tokens": args.max_image_tokens,
        "enable_thinking": False,
    }
    if args.max_new_tokens:
        generation_config["max_new_tokens"] = args.max_new_tokens
    result = run_cross_composition_evaluation(
        model=model,
        f0=f0,
        f1=f1,
        dataset=dataset,
        split=split,
        output_dir=args.output_dir,
        compositions=args.compositions,
        batch_size=args.batch_size,
        generation_config=generation_config,
        dataset_id_hash=dataset_id_hash,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
