"""Joint Shared+Task SoftSkill training with progressive decoupling."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from skillopt.config import flatten_config, is_structured, load_config
from skillopt.evaluation.gate import select_gate_score
from skillopt.softprefix.data import PrefixBatchCollator
from skillopt.softprefix.model import (
    SharedTaskSoftPrefixVisionLM,
    TaskSpecificMultiPrefixVisionLM,
)
from skillopt.softprefix.trainer import (
    SoftPrefixSettings,
    _batch_to_tensors,
    _build_dataloader,
    _build_dataset,
    _evaluate_prefix,
    _items_for_eval,
    _load_init_text,
    _set_seed,
)


@dataclass(slots=True)
class JointTask:
    name: str
    env: str
    config_path: str
    split_dir: str
    cfg: dict[str, Any]
    settings: SoftPrefixSettings
    init_text: str
    dataloader: Any
    train_loader: Any
    val_items: list[dict]
    test_items: list[dict]


def progressive_learning_rates(
    progress: float,
    *,
    shared_start: float,
    shared_min: float,
    shared_decay: float,
    task_max: float,
    task_growth: float,
) -> tuple[float, float]:
    """Exponentially decay shared LR while smoothly increasing task LR."""
    progress = max(0.0, min(float(progress), 1.0))
    shared_lr = max(float(shared_min), float(shared_start) * math.exp(-float(shared_decay) * progress))
    task_lr = float(task_max) * (1.0 - math.exp(-float(task_growth) * progress))
    return shared_lr, task_lr


def _load_task(
    spec: dict[str, str],
    *,
    model: SharedTaskSoftPrefixVisionLM,
    seed: int,
) -> JointTask:
    raw_cfg = load_config(spec["config"])
    cfg = flatten_config(raw_cfg) if is_structured(raw_cfg) else dict(raw_cfg)
    cfg["split_dir"] = spec["split_dir"]
    cfg["split_mode"] = "split_dir"
    cfg["seed"] = seed
    cfg["split_seed"] = seed
    cfg["out_root"] = ""
    settings = SoftPrefixSettings.from_dict(dict(raw_cfg.get("soft_prefix", {})))
    settings.inference_backend = "local_hf"
    dataloader = _build_dataloader(spec["env"], cfg, seed)
    dataloader.setup(cfg)
    train_items = _items_for_eval(dataloader, "train", int(cfg.get("train_size", 0) or 0), seed)
    dataset = _build_dataset(spec["env"], train_items, model, cfg, settings)
    collator = PrefixBatchCollator(model.tokenizer.pad_token_id)
    generator = model.torch.Generator()
    generator.manual_seed(seed)
    train_loader = model.torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    return JointTask(
        name=spec["name"],
        env=spec["env"],
        config_path=spec["config"],
        split_dir=spec["split_dir"],
        cfg=cfg,
        settings=settings,
        init_text=_load_init_text(settings.init_text_path or str(cfg.get("skill_init", ""))),
        dataloader=dataloader,
        train_loader=train_loader,
        val_items=_items_for_eval(dataloader, "valid_seen", int(cfg.get("sel_env_num", 0) or 0), seed),
        test_items=_items_for_eval(dataloader, "valid_unseen", int(cfg.get("test_env_num", 0) or 0), seed),
    )


def _save_checkpoint(torch, model, optimizer, path: str, *, epoch: int, global_step: int, history: list[dict]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
        },
        path,
    )


def _save_folded_prefixes(torch, model, tasks: list[JointTask], path: str) -> None:
    """Materialize Phi(P) so inference can discard all residual MLPs."""
    folded = {}
    with torch.no_grad():
        for task in tasks:
            model.set_active_task(task.name)
            folded[task.name] = model.active_prefix_embeddings().detach().cpu()
    torch.save(folded, path)


def _evaluate_tasks(
    model: SharedTaskSoftPrefixVisionLM,
    tasks: list[JointTask],
    *,
    out_root: str,
    stage: str,
    use_test: bool,
) -> tuple[float, dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, float]] = {}
    scores: list[float] = []
    for task in tasks:
        model.set_active_task(task.name)
        items = task.test_items if use_test else task.val_items
        hard, soft, _ = _evaluate_prefix(
            task.env,
            model,
            items,
            cfg=task.cfg,
            settings=task.settings,
            out_dir=os.path.join(out_root, "eval", stage, task.name),
            desc=f"  {stage} {task.name}",
        )
        gate_metric = str(task.cfg.get("gate_metric", "hard") or "hard")
        mixed_weight = float(task.cfg.get("gate_mixed_weight", 0.5) or 0.5)
        score = select_gate_score(hard, soft, gate_metric, mixed_weight)
        metrics[task.name] = {"hard": hard, "soft": soft, "score": score}
        scores.append(score)
    return sum(scores) / max(len(scores), 1), metrics


def train_shared_task_soft_prefix(
    *,
    task_specs: list[dict[str, str]],
    model_name: str,
    out_root: str,
    seed: int = 1,
    prefix_length: int = 32,
    num_epochs: int = 3,
    shared_lr_start: float = 1e-3,
    shared_lr_min: float = 5e-5,
    shared_decay: float = 3.0,
    task_lr_max: float = 1e-3,
    task_growth: float = 5.0,
    residual_bottleneck_size: int = 400,
    use_residual_reparameterization: bool = True,
) -> dict[str, Any]:
    """Jointly train a shared prefix and one task prefix for every benchmark."""
    if len(task_specs) < 2:
        raise ValueError("joint training requires at least two tasks")
    _set_seed(seed)
    os.makedirs(out_root, exist_ok=True)

    task_init_texts = {}
    for spec in task_specs:
        raw = load_config(spec["config"])
        settings = SoftPrefixSettings.from_dict(dict(raw.get("soft_prefix", {})))
        task_init_texts[spec["name"]] = _load_init_text(settings.init_text_path)
    shared_init_text = "\n\n".join(task_init_texts[spec["name"]] for spec in task_specs)

    model = SharedTaskSoftPrefixVisionLM(
        model_name,
        prefix_length=prefix_length,
        shared_init_text=shared_init_text,
        task_init_texts=task_init_texts,
        residual_bottleneck_size=residual_bottleneck_size,
        use_residual_reparameterization=use_residual_reparameterization,
        torch_dtype="auto",
        device="auto",
        trust_remote_code=True,
    )
    torch = model.torch
    tasks = [_load_task(spec, model=model, seed=seed) for spec in task_specs]
    cycles_per_epoch = max(len(task.train_loader) for task in tasks)
    total_steps = max(num_epochs * cycles_per_epoch, 1)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.shared_parameters(), "lr": shared_lr_start, "weight_decay": 0.0, "name": "shared"},
            {"params": model.task_parameters(), "lr": 0.0, "weight_decay": 0.0, "name": "task"},
        ]
    )

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_path = os.path.join(out_root, "best_shared_task.pt")
    latest_path = os.path.join(out_root, "latest_shared_task.pt")
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        started = time.time()
        iterators = {task.name: iter(task.train_loader) for task in tasks}
        task_losses = {task.name: [] for task in tasks}
        bar = tqdm(range(cycles_per_epoch), desc=f"Joint train {epoch}/{num_epochs}", unit="cycle")
        for _ in bar:
            progress = global_step / max(total_steps - 1, 1)
            shared_lr, task_lr = progressive_learning_rates(
                progress,
                shared_start=shared_lr_start,
                shared_min=shared_lr_min,
                shared_decay=shared_decay,
                task_max=task_lr_max,
                task_growth=task_growth,
            )
            optimizer.param_groups[0]["lr"] = shared_lr
            optimizer.param_groups[1]["lr"] = task_lr
            optimizer.zero_grad(set_to_none=True)
            for task in tasks:
                model.set_active_task(task.name)
                try:
                    batch = next(iterators[task.name])
                except StopIteration:
                    iterators[task.name] = iter(task.train_loader)
                    batch = next(iterators[task.name])
                tensor_batch = _batch_to_tensors(torch, batch, model.device)
                output = model.forward(tensor_batch)
                (output.loss / len(tasks)).backward()
                task_losses[task.name].append(float(output.loss.detach().cpu()))
            optimizer.step()
            global_step += 1
            bar.set_postfix(shared_lr=f"{shared_lr:.2e}", task_lr=f"{task_lr:.2e}")

        joint_score, val_metrics = _evaluate_tasks(
            model,
            tasks,
            out_root=out_root,
            stage=f"epoch_{epoch:02d}_valid_seen",
            use_test=False,
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "joint_score": joint_score,
            "validation": val_metrics,
            "mean_train_loss": {
                name: sum(values) / max(len(values), 1)
                for name, values in task_losses.items()
            },
            "shared_lr": optimizer.param_groups[0]["lr"],
            "task_lr": optimizer.param_groups[1]["lr"],
            "wall_time_s": round(time.time() - started, 1),
        }
        history.append(record)
        _save_checkpoint(torch, model, optimizer, latest_path, epoch=epoch, global_step=global_step, history=history)
        if joint_score > best_score:
            best_score = joint_score
            _save_checkpoint(torch, model, optimizer, best_path, epoch=epoch, global_step=global_step, history=history)
        with open(os.path.join(out_root, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)

    best_state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_state["model"])
    folded_path = os.path.join(out_root, "folded_prefixes.pt")
    _save_folded_prefixes(torch, model, tasks, folded_path)
    _, test_metrics = _evaluate_tasks(
        model,
        tasks,
        out_root=out_root,
        stage="best_valid_unseen",
        use_test=True,
    )
    summary = {
        "best_joint_score": best_score,
        "best_checkpoint": best_path,
        "latest_checkpoint": latest_path,
        "folded_prefixes": folded_path,
        "test": test_metrics,
        "history": history,
        "prefix_length": prefix_length,
        "effective_prefix_length": prefix_length * 2,
        "residual_bottleneck_size": residual_bottleneck_size,
        "use_residual_reparameterization": use_residual_reparameterization,
        "seed": seed,
    }
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def train_joint_task_multi_prefix(
    *,
    task_specs: list[dict[str, str]],
    model_name: str,
    out_root: str,
    seed: int = 1,
    prefix_length: int = 16,
    num_soft_skills: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
    residual_bottleneck_size: int = 400,
    use_residual_reparameterization: bool = True,
) -> dict[str, Any]:
    """Jointly schedule tasks while keeping an independent multi-prefix per task."""
    if len(task_specs) < 2:
        raise ValueError("joint training requires at least two tasks")
    _set_seed(seed)
    os.makedirs(out_root, exist_ok=True)

    task_init_texts = {}
    for spec in task_specs:
        raw = load_config(spec["config"])
        settings = SoftPrefixSettings.from_dict(dict(raw.get("soft_prefix", {})))
        task_init_texts[spec["name"]] = _load_init_text(settings.init_text_path)

    model = TaskSpecificMultiPrefixVisionLM(
        model_name,
        prefix_length=prefix_length,
        num_soft_skills=num_soft_skills,
        task_init_texts=task_init_texts,
        residual_bottleneck_size=residual_bottleneck_size,
        use_residual_reparameterization=use_residual_reparameterization,
        torch_dtype="auto",
        device="auto",
        trust_remote_code=True,
    )
    torch = model.torch
    tasks = [_load_task(spec, model=model, seed=seed) for spec in task_specs]
    cycles_per_epoch = max(len(task.train_loader) for task in tasks)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_path = os.path.join(out_root, "best_joint_2x16.pt")
    latest_path = os.path.join(out_root, "latest_joint_2x16.pt")
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        started = time.time()
        iterators = {task.name: iter(task.train_loader) for task in tasks}
        task_losses = {task.name: [] for task in tasks}
        bar = tqdm(range(cycles_per_epoch), desc=f"Joint 2x16 train {epoch}/{num_epochs}", unit="cycle")
        for _ in bar:
            optimizer.zero_grad(set_to_none=True)
            for task in tasks:
                model.set_active_task(task.name)
                try:
                    batch = next(iterators[task.name])
                except StopIteration:
                    iterators[task.name] = iter(task.train_loader)
                    batch = next(iterators[task.name])
                tensor_batch = _batch_to_tensors(torch, batch, model.device)
                output = model.forward(tensor_batch)
                (output.loss / len(tasks)).backward()
                task_losses[task.name].append(float(output.loss.detach().cpu()))
            optimizer.step()
            global_step += 1
            bar.set_postfix(lr=f"{learning_rate:.2e}")

        joint_score, val_metrics = _evaluate_tasks(
            model,
            tasks,
            out_root=out_root,
            stage=f"epoch_{epoch:02d}_valid_seen",
            use_test=False,
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "joint_score": joint_score,
            "validation": val_metrics,
            "mean_train_loss": {
                name: sum(values) / max(len(values), 1)
                for name, values in task_losses.items()
            },
            "learning_rate": learning_rate,
            "wall_time_s": round(time.time() - started, 1),
        }
        history.append(record)
        _save_checkpoint(torch, model, optimizer, latest_path, epoch=epoch, global_step=global_step, history=history)
        if joint_score > best_score:
            best_score = joint_score
            _save_checkpoint(torch, model, optimizer, best_path, epoch=epoch, global_step=global_step, history=history)
        with open(os.path.join(out_root, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)

    best_state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_state["model"])
    folded_path = os.path.join(out_root, "folded_prefixes.pt")
    _save_folded_prefixes(torch, model, tasks, folded_path)
    _, test_metrics = _evaluate_tasks(
        model,
        tasks,
        out_root=out_root,
        stage="best_valid_unseen",
        use_test=True,
    )
    summary = {
        "method": "joint_task_specific_multi_prefix",
        "best_joint_score": best_score,
        "best_checkpoint": best_path,
        "latest_checkpoint": latest_path,
        "folded_prefixes": folded_path,
        "test": test_metrics,
        "history": history,
        "prefix_length": prefix_length,
        "num_soft_skills": num_soft_skills,
        "effective_prefix_length": prefix_length * num_soft_skills,
        "residual_bottleneck_size": residual_bottleneck_size,
        "use_residual_reparameterization": use_residual_reparameterization,
        "seed": seed,
    }
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
