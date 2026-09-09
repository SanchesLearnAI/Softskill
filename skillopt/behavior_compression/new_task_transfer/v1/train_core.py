"""Minimal deterministic target-only training core for transfer stage 3A."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Sequence


VALID_TASKS = {"chartqa", "drop"}
VALID_CONDITIONS = {"f0", "f1", "i32"}


@dataclass(frozen=True)
class MinimalTrainConfig:
    task: str
    condition: str
    seed: int
    source_checkpoint: str = ""
    output_dir: str = ""
    max_optimizer_steps: int = 150
    learning_rate: float = 1e-3
    gradient_accumulation_steps: int = 1
    batch_size: int = 1

    def validated(self) -> "MinimalTrainConfig":
        task = str(self.task).strip().lower()
        condition = str(self.condition).strip().lower()
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
        if condition not in VALID_CONDITIONS:
            raise ValueError(f"condition must be one of {sorted(VALID_CONDITIONS)}")
        source = str(self.source_checkpoint or "").strip()
        if condition == "f1" and not source:
            raise ValueError("F1 requires source_checkpoint")
        if condition != "f1" and source:
            raise ValueError("only F1 accepts source_checkpoint")
        if int(self.max_optimizer_steps) < 1:
            raise ValueError("max_optimizer_steps must be positive")
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be positive")
        if int(self.gradient_accumulation_steps) < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be positive")
        return MinimalTrainConfig(
            task=task,
            condition=condition,
            seed=int(self.seed),
            source_checkpoint=source,
            output_dir=str(self.output_dir),
            max_optimizer_steps=int(self.max_optimizer_steps),
            learning_rate=float(self.learning_rate),
            gradient_accumulation_steps=int(self.gradient_accumulation_steps),
            batch_size=int(self.batch_size),
        )


@dataclass(frozen=True)
class DeterministicBatchSchedule:
    index_batches: tuple[tuple[int, ...], ...]
    sample_id_batches: tuple[tuple[str, ...], ...]
    sample_order_hash: str


def _schedule_seed(task: str, seed: int) -> int:
    payload = f"new_task_transfer_v1:{task}:{int(seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_deterministic_batch_schedule(
    sample_ids: Sequence[str],
    *,
    task: str,
    seed: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_optimizer_steps: int,
) -> DeterministicBatchSchedule:
    """Build all microbatches without consulting condition or global RNG state."""
    task = str(task).strip().lower()
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
    ids = [str(sample_id) for sample_id in sample_ids]
    if not ids:
        raise ValueError("training split must contain at least one sample")
    if int(batch_size) < 1 or int(gradient_accumulation_steps) < 1:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    if int(max_optimizer_steps) < 1:
        raise ValueError("max_optimizer_steps must be positive")

    required_microbatches = int(max_optimizer_steps) * int(
        gradient_accumulation_steps
    )
    rng = random.Random(_schedule_seed(task, seed))
    index_batches: list[tuple[int, ...]] = []
    while len(index_batches) < required_microbatches:
        epoch_indices = list(range(len(ids)))
        rng.shuffle(epoch_indices)
        for start in range(0, len(epoch_indices), int(batch_size)):
            index_batches.append(tuple(epoch_indices[start : start + int(batch_size)]))
            if len(index_batches) == required_microbatches:
                break
    sample_id_batches = tuple(
        tuple(ids[index] for index in batch) for batch in index_batches
    )
    serialized = json.dumps(
        sample_id_batches,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return DeterministicBatchSchedule(
        index_batches=tuple(index_batches),
        sample_id_batches=sample_id_batches,
        sample_order_hash=hashlib.sha256(serialized).hexdigest(),
    )


def _batch_to_device(torch_module: Any, batch: dict[str, Any], device: Any) -> dict:
    converted: dict[str, Any] = {}
    for name, value in batch.items():
        if hasattr(value, "to"):
            converted[name] = value.to(device)
        elif isinstance(value, list):
            converted[name] = torch_module.as_tensor(value, device=device)
        else:
            converted[name] = value
    return converted


def supervised_token_count(torch_module: Any, batch: dict[str, Any]) -> int:
    labels = batch.get("labels")
    if labels is None:
        return 0
    labels = labels if hasattr(labels, "ne") else torch_module.as_tensor(labels)
    shifted = torch_module.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    return int(shifted.ne(-100).sum().item())


def run_minimal_training(
    *,
    config: MinimalTrainConfig,
    model: Any,
    train_dataset: Any,
    collator: Any,
) -> dict[str, Any]:
    """Run exactly ``max_optimizer_steps`` with token-weighted accumulation."""
    config = config.validated()
    torch_module = model.torch
    if any(parameter.requires_grad for parameter in model.model.parameters()):
        raise RuntimeError("base model must be completely frozen")
    if bool(getattr(model, "use_residual_reparameterization", False)):
        raise RuntimeError("stage 3A requires residual disabled")
    if int(model.active_prefix_embeddings().shape[0]) != 32:
        raise RuntimeError("effective prefix length must be 32")
    if str(model.condition).strip().lower() != config.condition:
        raise RuntimeError("configured condition does not match the prefix model")
    if str(model.task_name).strip().lower() != config.task:
        raise RuntimeError("configured task does not match the prefix model")
    if callable(getattr(model, "validate_invariants", None)):
        model.validate_invariants()

    trainable = list(model.trainable_parameters())
    if not trainable or any(not parameter.requires_grad for parameter in trainable):
        raise RuntimeError("trainable_parameters must contain only trainable prefixes")
    prefix_parameter_ids = {
        id(parameter)
        for parameter in model.prefix_parameters.parameters()
        if parameter.requires_grad
    }
    if {id(parameter) for parameter in trainable} != prefix_parameter_ids:
        raise RuntimeError("optimizer parameters must be exactly the allowed prefix parameters")

    sample_ids = [str(item["id"]) for item in train_dataset.items]
    schedule = build_deterministic_batch_schedule(
        sample_ids,
        task=config.task,
        seed=config.seed,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_optimizer_steps=config.max_optimizer_steps,
    )
    optimizer = torch_module.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    optimizer_steps = 0
    microbatches_consumed = 0
    losses: list[float] = []
    model.model.eval()

    for step_index in range(config.max_optimizer_steps):
        start = step_index * config.gradient_accumulation_steps
        index_batches = schedule.index_batches[
            start : start + config.gradient_accumulation_steps
        ]
        raw_batches = [
            collator([train_dataset[index] for index in index_batch])
            for index_batch in index_batches
        ]
        token_counts = [
            supervised_token_count(torch_module, batch) for batch in raw_batches
        ]
        if any(count < 1 for count in token_counts):
            raise RuntimeError("every microbatch must contain supervised answer tokens")
        total_tokens = sum(token_counts)

        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        for raw_batch, token_count in zip(raw_batches, token_counts):
            batch = _batch_to_device(torch_module, raw_batch, model.device)
            output = model.forward(batch)
            if output.loss is None:
                raise RuntimeError("target-only model returned no NTP loss")
            scale = token_count / total_tokens
            (output.loss * scale).backward()
            weighted_loss += float(output.loss.detach().cpu()) * scale
            microbatches_consumed += 1
        optimizer.step()
        optimizer_steps += 1
        losses.append(weighted_loss)

    if optimizer_steps != config.max_optimizer_steps:
        raise RuntimeError("optimizer step count differs from max_optimizer_steps")
    expected_microbatches = (
        config.max_optimizer_steps * config.gradient_accumulation_steps
    )
    if microbatches_consumed != expected_microbatches:
        raise RuntimeError("gradient accumulation did not consume the expected microbatches")
    return {
        "task": config.task,
        "condition": config.condition,
        "seed": config.seed,
        "optimizer_steps": optimizer_steps,
        "microbatches_consumed": microbatches_consumed,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "sample_order_hash": schedule.sample_order_hash,
        "sample_occurrences": sum(len(batch) for batch in schedule.sample_id_batches),
        "losses": losses,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": 0.0,
            "lr_schedule": "constant",
        },
    }
