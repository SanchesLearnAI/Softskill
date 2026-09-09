#!/usr/bin/env python3
"""One-cycle GPU smoke test for the joint Shared+Task model."""
from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skillopt.config import load_config
from skillopt.softprefix.model import SharedTaskSoftPrefixVisionLM
from skillopt.softprefix.multitask_trainer import _load_task
from skillopt.softprefix.trainer import SoftPrefixSettings, _batch_to_tensors, _load_init_text


TASK_SPECS = [
    {"name": "searchqa", "env": "searchqa", "config": "configs/searchqa/soft_prefix_skill_section.yaml", "split_dir": "data/searchqa_split"},
    {"name": "livemath", "env": "livemathematicianbench", "config": "configs/livemathematicianbench/soft_prefix_bs4_acc2.yaml", "split_dir": "data/livemathematicianbench_split"},
    {"name": "docvqa", "env": "docvqa", "config": "configs/docvqa/soft_prefix.yaml", "split_dir": "data/docvqa/splits"},
]


def main() -> None:
    init_texts = {}
    for spec in TASK_SPECS:
        raw = load_config(spec["config"])
        settings = SoftPrefixSettings.from_dict(dict(raw["soft_prefix"]))
        init_texts[spec["name"]] = _load_init_text(settings.init_text_path)
    model = SharedTaskSoftPrefixVisionLM(
        "Qwen/Qwen3.5-4B",
        prefix_length=32,
        shared_init_text="\n\n".join(init_texts.values()),
        task_init_texts=init_texts,
        trust_remote_code=True,
    )
    tasks = [_load_task(spec, model=model, seed=1) for spec in TASK_SPECS]
    model.model.train(False)
    for parameter in model.trainable_parameters():
        parameter.grad = None
    losses = {}
    for task in tasks:
        model.set_active_task(task.name)
        batch = next(iter(task.train_loader))
        tensor_batch = _batch_to_tensors(model.torch, batch, model.device)
        output = model.forward(tensor_batch)
        (output.loss / len(tasks)).backward()
        losses[task.name] = float(output.loss.detach().cpu())
    assert model.prefix_embeddings.grad is not None
    for name, parameter in model.task_prefix_embeddings.items():
        assert parameter.grad is not None, f"missing gradient for {name}"
    print({"losses": losses, "shared_grad": float(model.prefix_embeddings.grad.float().norm().cpu())})


if __name__ == "__main__":
    main()
