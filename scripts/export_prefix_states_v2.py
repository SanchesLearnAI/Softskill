#!/usr/bin/env python3
"""Backfill four-state prefix artifacts from an existing v2 checkpoint.

This command loads the frozen base model only to reproduce the exact Markdown
embedding initialization.  It does not load datasets, train, or evaluate.
Future v2 training runs save the same artifacts automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skillopt.softprefix.multitask_v2 import (
    JointV2SoftPrefixVisionLM,
    _load_task_markdown,
    initialization_kwargs_from_checkpoint_state,
    save_initial_prefix_snapshot,
    save_prefix_state_artifact,
)


DEFAULT_TASKS = (
    ("searchqa", "configs/searchqa/soft_prefix_2x16.yaml"),
    ("livemath", "configs/livemathematicianbench/soft_prefix_2x16.yaml"),
    ("docvqa", "configs/docvqa/soft_prefix_2x16.yaml"),
)


def _parse_task_specs(values: list[str]) -> list[dict[str, str]]:
    pairs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--task must be NAME=CONFIG, got {value!r}")
        name, config = value.split("=", 1)
        if not name.strip() or not config.strip():
            raise ValueError(f"--task must be NAME=CONFIG, got {value!r}")
        pairs.append((name.strip(), config.strip()))
    if not pairs:
        pairs = list(DEFAULT_TASKS)
    names = [name for name, _ in pairs]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("at least two uniquely named tasks are required")
    return [{"name": name, "config": config} for name, config in pairs]


def _load_checkpoint(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Existing best_v2.pt")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="NAME=CONFIG",
        help="Repeat for each task; defaults to SearchQA, LiveMath, and DocVQA",
    )
    parser.add_argument("--output", help="Defaults to CHECKPOINT_DIR/prefix_states.pt")
    parser.add_argument(
        "--initial-output",
        help="Defaults to CHECKPOINT_DIR/initial_prefix_states.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_path = Path(args.output).resolve() if args.output else checkpoint_path.parent / "prefix_states.pt"
    initial_path = (
        Path(args.initial_output).resolve()
        if args.initial_output
        else checkpoint_path.parent / "initial_prefix_states.pt"
    )
    for path in (output_path, initial_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    checkpoint = _load_checkpoint(torch, checkpoint_path)
    state = checkpoint.get("model")
    if not isinstance(state, dict) or state.get("protocol_version") != "v2":
        raise ValueError("checkpoint does not contain a v2 model state")

    task_specs = _parse_task_specs(list(args.task))
    task_init_texts = _load_task_markdown(task_specs)
    checkpoint_tasks = set(state.get("task_prefix_embeddings", {}))
    if checkpoint_tasks != set(task_init_texts):
        raise ValueError(
            f"task mismatch: checkpoint={sorted(checkpoint_tasks)}, "
            f"requested={sorted(task_init_texts)}"
        )

    initialization_kwargs = initialization_kwargs_from_checkpoint_state(
        state,
        fallback_task_init_texts=task_init_texts,
    )
    model = JointV2SoftPrefixVisionLM(
        args.model_name,
        layout=str(state["layout"]),
        prefix_length=int(state["prefix_length"]),
        **initialization_kwargs,
        residual_bottleneck_size=int(state["residual_bottleneck_size"]),
        use_residual_reparameterization=bool(state["use_residual_reparameterization"]),
        residual_mode=str(state["residual_mode"]),
        residual_init_seed=int(state["residual_init_seed"]),
        torch_dtype=args.torch_dtype,
        device=args.device,
        trust_remote_code=True,
    )
    task_names = [spec["name"] for spec in task_specs]
    initial_snapshot = save_initial_prefix_snapshot(
        torch,
        model,
        task_names,
        str(initial_path),
    )
    model.load_state_dict(state)
    artifact = save_prefix_state_artifact(
        torch,
        model,
        task_names,
        initial_snapshot,
        str(output_path),
        source_checkpoint=str(checkpoint_path),
    )
    print(json.dumps({
        "status": "ok",
        "checkpoint": str(checkpoint_path),
        "initial_prefix_states": str(initial_path),
        "prefix_states": str(output_path),
        "layout": artifact["layout"],
        "tasks": artifact["task_order"],
        "invariants": artifact["invariants"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
