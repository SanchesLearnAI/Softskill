#!/usr/bin/env python3
"""Train one shared SoftSkill plus one task SoftSkill across three benchmarks."""
from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skillopt.softprefix.multitask_trainer import train_shared_task_soft_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint Shared+Task SoftSkill trainer")
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prefix_length", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--shared_lr_start", type=float, default=1e-3)
    parser.add_argument("--shared_lr_min", type=float, default=5e-5)
    parser.add_argument("--shared_decay", type=float, default=3.0)
    parser.add_argument("--task_lr_max", type=float, default=1e-3)
    parser.add_argument("--task_growth", type=float, default=5.0)
    parser.add_argument("--residual_bottleneck_size", type=int, default=400)
    parser.add_argument("--disable_residual_reparameterization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_specs = [
        {
            "name": "searchqa",
            "env": "searchqa",
            "config": "configs/searchqa/soft_prefix_skill_section.yaml",
            "split_dir": "data/searchqa_split",
        },
        {
            "name": "livemath",
            "env": "livemathematicianbench",
            "config": "configs/livemathematicianbench/soft_prefix_bs4_acc2.yaml",
            "split_dir": "data/livemathematicianbench_split",
        },
        {
            "name": "docvqa",
            "env": "docvqa",
            "config": "configs/docvqa/soft_prefix.yaml",
            "split_dir": "data/docvqa/splits",
        },
    ]
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "joint_config.json"), "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "tasks": task_specs}, handle, ensure_ascii=False, indent=2)
    summary = train_shared_task_soft_prefix(
        task_specs=task_specs,
        model_name=args.model_name,
        out_root=os.path.abspath(args.out_root),
        seed=args.seed,
        prefix_length=args.prefix_length,
        num_epochs=args.num_epochs,
        shared_lr_start=args.shared_lr_start,
        shared_lr_min=args.shared_lr_min,
        shared_decay=args.shared_decay,
        task_lr_max=args.task_lr_max,
        task_growth=args.task_growth,
        residual_bottleneck_size=args.residual_bottleneck_size,
        use_residual_reparameterization=not args.disable_residual_reparameterization,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
