"""Bounded early-sample generation check for the trained DROP I32 prefix."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from skillopt.behavior_compression.new_task_transfer.v1.evaluation import (
    evaluate_transfer_predictions,
)
from skillopt.behavior_compression.new_task_transfer.v1.model import (
    TargetOnlyTransferSoftPrefixVisionLM,
)
from skillopt.envs.drop.data import select_drop_transfer_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--num_samples", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("num_samples must be positive")
    model = TargetOnlyTransferSoftPrefixVisionLM(
        args.model_name,
        task_name="drop",
        condition="i32",
        source_checkpoint_path=None,
        torch_dtype="auto",
        device="auto",
        trust_remote_code=True,
    )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    try:
        checkpoint = model.torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = model.torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("task") != "drop" or checkpoint.get("condition") != "i32":
        raise RuntimeError("diagnostic checkpoint identity mismatch")
    if int(checkpoint.get("optimizer_step", -1)) != 150:
        raise RuntimeError("diagnostic requires the step-150 final checkpoint")
    model.load_state_dict(checkpoint["prefix_state"])
    splits, _manifest = select_drop_transfer_splits(args.data_root, seed=1)
    started = time.perf_counter()
    result = evaluate_transfer_predictions(
        model=model,
        task="drop",
        dataset=splits["validation"][: args.num_samples],
        split="validation",
        batch_size=1,
        generation_config={
            "max_prompt_tokens": 8192,
            "max_new_tokens": 128,
            "temperature": 0.0,
            "enable_thinking": False,
        },
        predictions_path=args.predictions_path,
    )
    result["wall_elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
