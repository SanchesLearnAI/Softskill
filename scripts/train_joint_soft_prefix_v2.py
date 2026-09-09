#!/usr/bin/env python3
"""Run one corrected v2 joint SoftSkill experiment cell."""
from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skillopt.softprefix.multitask_v2 import (
    INIT_BEHAVIOR_MARKDOWN,
    INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD,
    INIT_LEGACY_TASK_BLOCKS,
    LR_BEHAVIOR_TWO_STAGE,
    LR_PROGRESSIVE,
    LR_UNIFORM,
    LAYOUT_INDEPENDENT,
    LAYOUT_SHARED_TASK,
    RESIDUAL_BRANCH_DECOUPLED,
    RESIDUAL_GLOBAL_MATCHED,
    train_joint_soft_prefix_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrected v2 joint SoftSkill trainer")
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--layout", choices=[LAYOUT_INDEPENDENT, LAYOUT_SHARED_TASK], required=True)
    parser.add_argument(
        "--initialization_mode",
        choices=[
            INIT_LEGACY_TASK_BLOCKS,
            INIT_BEHAVIOR_MARKDOWN,
            INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD,
        ],
        default=INIT_LEGACY_TASK_BLOCKS,
    )
    parser.add_argument(
        "--shared_behavior_path",
        default="skillopt/behavior_compression/v1/shared_behavior.md",
    )
    parser.add_argument(
        "--behavior_provenance_path",
        default="skillopt/behavior_compression/v1/provenance.json",
    )
    parser.add_argument(
        "--behavior_tokenizer_audit_path",
        default="skillopt/behavior_compression/v1/tokenizer_audit.json",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prefix_length", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--expected_optimizer_steps", type=int)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr_schedule",
        choices=[LR_UNIFORM, LR_PROGRESSIVE, LR_BEHAVIOR_TWO_STAGE],
        default=LR_UNIFORM,
    )
    parser.add_argument("--shared_lr_start", type=float, default=1e-3)
    parser.add_argument("--shared_lr_min", type=float, default=5e-5)
    parser.add_argument("--shared_decay", type=float, default=3.0)
    parser.add_argument("--task_lr_max", type=float, default=1e-3)
    parser.add_argument("--task_growth", type=float, default=5.0)
    parser.add_argument("--behavior_warmup_fraction", type=float, default=0.2)
    parser.add_argument("--behavior_shared_warmup_lr", type=float, default=1e-3)
    parser.add_argument("--behavior_shared_joint_lr", type=float, default=1e-4)
    parser.add_argument("--behavior_task_joint_lr", type=float, default=1e-3)
    parser.add_argument("--behavior_residual_joint_lr", type=float, default=1e-3)
    parser.add_argument("--residual_bottleneck_size", type=int, default=400)
    parser.add_argument("--residual_reparameterization", action="store_true")
    parser.add_argument(
        "--residual_mode",
        choices=[RESIDUAL_GLOBAL_MATCHED, RESIDUAL_BRANCH_DECOUPLED],
        default=RESIDUAL_GLOBAL_MATCHED,
    )
    parser.add_argument("--residual_init_seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_specs = [
        {
            "name": "searchqa",
            "env": "searchqa",
            "config": "configs/searchqa/soft_prefix_2x16.yaml",
            "split_dir": "data/searchqa_split",
            "behavior_init_path": "skillopt/behavior_compression/v1/searchqa_behavior.md",
        },
        {
            "name": "livemath",
            "env": "livemathematicianbench",
            "config": "configs/livemathematicianbench/soft_prefix_2x16.yaml",
            "split_dir": "data/livemathematicianbench_split",
            "behavior_init_path": "skillopt/behavior_compression/v1/livemath_behavior.md",
        },
        {
            "name": "docvqa",
            "env": "docvqa",
            "config": "configs/docvqa/soft_prefix_2x16.yaml",
            "split_dir": "data/docvqa/splits",
            "behavior_init_path": "skillopt/behavior_compression/v1/docvqa_behavior.md",
        },
    ]
    os.makedirs(args.out_root, exist_ok=True)
    manifest = {
        "protocol_version": "v2",
        "args": vars(args),
        "tasks": task_specs,
    }
    with open(os.path.join(args.out_root, "joint_config.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    summary = train_joint_soft_prefix_v2(
        task_specs=task_specs,
        model_name=args.model_name,
        out_root=os.path.abspath(args.out_root),
        layout=args.layout,
        initialization_mode=args.initialization_mode,
        shared_behavior_path=args.shared_behavior_path,
        behavior_provenance_path=args.behavior_provenance_path,
        behavior_tokenizer_audit_path=args.behavior_tokenizer_audit_path,
        seed=args.seed,
        prefix_length=args.prefix_length,
        num_epochs=args.num_epochs,
        expected_optimizer_steps=args.expected_optimizer_steps,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        shared_lr_start=args.shared_lr_start,
        shared_lr_min=args.shared_lr_min,
        shared_decay=args.shared_decay,
        task_lr_max=args.task_lr_max,
        task_growth=args.task_growth,
        behavior_warmup_fraction=args.behavior_warmup_fraction,
        behavior_shared_warmup_lr=args.behavior_shared_warmup_lr,
        behavior_shared_joint_lr=args.behavior_shared_joint_lr,
        behavior_task_joint_lr=args.behavior_task_joint_lr,
        behavior_residual_joint_lr=args.behavior_residual_joint_lr,
        residual_bottleneck_size=args.residual_bottleneck_size,
        use_residual_reparameterization=args.residual_reparameterization,
        residual_mode=args.residual_mode,
        residual_init_seed=args.residual_init_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
