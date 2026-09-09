#!/usr/bin/env python3
"""Real-model optimizer-step smoke tests for the corrected v2 protocol."""
from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skillopt.config import load_config
from skillopt.softprefix.multitask_trainer import _load_task
from skillopt.softprefix.multitask_v2 import (
    INIT_BEHAVIOR_MARKDOWN,
    INIT_LEGACY_TASK_BLOCKS,
    LR_BEHAVIOR_TWO_STAGE,
    LR_PROGRESSIVE,
    LR_UNIFORM,
    LAYOUT_INDEPENDENT,
    LAYOUT_SHARED_TASK,
    RESIDUAL_BRANCH_DECOUPLED,
    RESIDUAL_GLOBAL_MATCHED,
    JointV2SoftPrefixVisionLM,
    _load_behavior_markdown,
    _build_optimizer,
    _task_accumulation,
    set_behavior_training_stage,
    batch_row_count,
    supervised_token_count,
    token_weighted_microbatch_scales,
)
from skillopt.softprefix.trainer import (
    SoftPrefixSettings,
    _batch_to_tensors,
    _load_init_text,
    _set_seed,
)


TASK_SPECS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layout", choices=[LAYOUT_INDEPENDENT, LAYOUT_SHARED_TASK], default=LAYOUT_SHARED_TASK)
    parser.add_argument(
        "--initialization_mode",
        choices=[INIT_LEGACY_TASK_BLOCKS, INIT_BEHAVIOR_MARKDOWN],
        default=INIT_LEGACY_TASK_BLOCKS,
    )
    parser.add_argument(
        "--shared_behavior_path",
        default="skillopt/behavior_compression/v1/shared_behavior.md",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--progressive", action="store_true")
    parser.add_argument("--behavior_two_stage", action="store_true")
    parser.add_argument("--out_root", default="outputs/behavior_compression/v1/smoke")
    return parser.parse_args()


def _run_joint_step(model, tasks, optimizer):
    optimizer.zero_grad(set_to_none=True)
    losses = {}
    samples = {}
    tokens = {}
    for task in tasks:
        model.set_active_task(task.name)
        iterator = iter(task.train_loader)
        batches = [next(iterator) for _ in range(_task_accumulation(task))]
        active_counts = [supervised_token_count(model.torch, batch) for batch in batches]
        scales = token_weighted_microbatch_scales(active_counts, num_tasks=len(tasks))
        weighted_loss = 0.0
        for batch, active_count, scale in zip(batches, active_counts, scales):
            tensor_batch = _batch_to_tensors(model.torch, batch, model.device)
            output = model.forward(tensor_batch)
            (output.loss * scale).backward()
            weighted_loss += float(output.loss.detach().cpu()) * active_count
        losses[task.name] = weighted_loss / sum(active_counts)
        samples[task.name] = sum(batch_row_count(batch) for batch in batches)
        tokens[task.name] = sum(active_counts)
    optimizer.step()
    return {"losses": losses, "samples": samples, "supervised_tokens": tokens}


def main() -> None:
    args = parse_args()
    if args.progressive and args.behavior_two_stage:
        raise ValueError("choose at most one non-uniform LR schedule")
    if args.progressive and args.layout != LAYOUT_SHARED_TASK:
        raise ValueError("progressive smoke testing requires shared16_task16")
    if args.behavior_two_stage:
        if args.layout != LAYOUT_SHARED_TASK:
            raise ValueError("behavior two-stage smoke requires shared16_task16")
        if args.initialization_mode != INIT_BEHAVIOR_MARKDOWN:
            raise ValueError("behavior two-stage smoke requires behavior_markdown")
        if args.residual:
            raise ValueError("behavior v1 smoke requires residual=False")
    _set_seed(args.seed)
    init_texts = {}
    for spec in TASK_SPECS:
        raw = load_config(spec["config"])
        settings = SoftPrefixSettings.from_dict(dict(raw["soft_prefix"]))
        init_texts[spec["name"]] = _load_init_text(settings.init_text_path)
    residual_mode = (
        RESIDUAL_BRANCH_DECOUPLED
        if args.progressive and args.residual
        else RESIDUAL_GLOBAL_MATCHED
    )
    shared_behavior_text = ""
    task_behavior_texts = {}
    if args.initialization_mode == INIT_BEHAVIOR_MARKDOWN:
        shared_behavior_text, task_behavior_texts = _load_behavior_markdown(
            TASK_SPECS,
            shared_behavior_path=args.shared_behavior_path,
        )
    model = JointV2SoftPrefixVisionLM(
        args.model_name,
        layout=args.layout,
        prefix_length=16,
        task_init_texts=init_texts,
        initialization_mode=args.initialization_mode,
        shared_behavior_text=shared_behavior_text,
        task_behavior_texts=task_behavior_texts,
        use_residual_reparameterization=args.residual,
        residual_mode=residual_mode,
        residual_init_seed=args.seed + 1_000_003,
        trust_remote_code=True,
    )
    tasks = [_load_task(spec, model=model, seed=args.seed) for spec in TASK_SPECS]
    lr_schedule = (
        LR_BEHAVIOR_TWO_STAGE
        if args.behavior_two_stage
        else (LR_PROGRESSIVE if args.progressive else LR_UNIFORM)
    )
    optimizer = _build_optimizer(
        model.torch,
        model,
        learning_rate=1e-3,
        lr_schedule=lr_schedule,
        shared_lr_start=1e-3,
    )
    stage_events = []
    if args.behavior_two_stage:
        before_tasks = {
            name: value.detach().clone()
            for name, value in model.task_prefix_embeddings.items()
        }
        event = set_behavior_training_stage(
            model,
            optimizer,
            stage="shared_warmup",
        )
        event["global_step"] = 0
        stage_events.append(event)
        warmup_result = _run_joint_step(model, tasks, optimizer)
        for name, value in model.task_prefix_embeddings.items():
            assert model.torch.equal(before_tasks[name], value.detach())

        before_tasks = {
            name: value.detach().clone()
            for name, value in model.task_prefix_embeddings.items()
        }
        event = set_behavior_training_stage(
            model,
            optimizer,
            stage="shared_task_joint",
        )
        event["global_step"] = 1
        stage_events.append(event)
        joint_result = _run_joint_step(model, tasks, optimizer)
        for name, value in model.task_prefix_embeddings.items():
            assert not model.torch.equal(before_tasks[name], value.detach())
        results = {"shared_warmup": warmup_result, "shared_task_joint": joint_result}
        global_step = 2
    else:
        results = {"single_step": _run_joint_step(model, tasks, optimizer)}
        global_step = 1

    for task in tasks:
        model.set_active_task(task.name)
        assert tuple(model.active_prefix_embeddings().shape[:1]) == (32,)
    assert all(parameter.grad is None for parameter in model.model.parameters())
    os.makedirs(args.out_root, exist_ok=True)
    checkpoint_path = os.path.join(args.out_root, "smoke_checkpoint.pt")
    model.torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "stage_events": stage_events,
            "results": results,
        },
        checkpoint_path,
    )
    saved = model.torch.load(checkpoint_path, map_location="cpu")
    assert saved["global_step"] == global_step
    report = {
        "layout": args.layout,
        "lr_schedule": lr_schedule,
        "residual_mode": residual_mode,
        "results": results,
        "stage_events": stage_events,
        "checkpoint": os.path.abspath(checkpoint_path),
        "initialization": model.initialization_audit,
    }
    with open(os.path.join(args.out_root, "smoke_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
