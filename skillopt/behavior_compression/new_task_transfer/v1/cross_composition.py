"""Checkpoint-only Shared16/Task16 cross-composition evaluation.

This module never trains a prefix.  It validates completed F0/F1 checkpoints,
constructs the two native and two crossed 32-token prefixes, and evaluates each
composition through the existing task evaluator.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import (
    file_sha256,
    tensor_sha256,
)
from skillopt.behavior_compression.new_task_transfer.v1.evaluation import (
    evaluate_transfer_predictions,
)
from skillopt.behavior_compression.new_task_transfer.v1.train_state import (
    PROTOCOL_VERSION as TRAIN_CHECKPOINT_PROTOCOL,
)


FINAL_OPTIMIZER_STEP = 150
PREFIX_BLOCK_LENGTH = 16
EFFECTIVE_PREFIX_LENGTH = 32

RAW_SHARED_F0_TASK = "raw_shared_f0_task"
TRANSFERRED_SHARED_F1_TASK = "transferred_shared_f1_task"
TRANSFERRED_SHARED_F0_TASK = "transferred_shared_f0_task"
RAW_SHARED_F1_TASK = "raw_shared_f1_task"

COMPOSITION_SOURCES = {
    RAW_SHARED_F0_TASK: ("f0", "f0"),
    TRANSFERRED_SHARED_F1_TASK: ("f1", "f1"),
    TRANSFERRED_SHARED_F0_TASK: ("f1", "f0"),
    RAW_SHARED_F1_TASK: ("f0", "f1"),
}
ALL_COMPOSITIONS = tuple(COMPOSITION_SOURCES)
# The native F0/F1 compositions already have formal predictions. Default to
# only the missing crossed pairs so an accidental CLI run does not regenerate
# known baselines; callers can still request all four explicitly.
DEFAULT_COMPOSITIONS = (
    TRANSFERRED_SHARED_F0_TASK,
    RAW_SHARED_F1_TASK,
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _torch_load(torch_module: Any, path: Path) -> dict[str, Any]:
    try:
        value = torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch_module.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint must be a dictionary: {path}")
    return value


def _validate_tensor(
    torch_module: Any,
    value: Any,
    *,
    name: str,
    hidden_size: int | None,
) -> int:
    if not torch_module.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.dim() != 2 or int(value.shape[0]) != PREFIX_BLOCK_LENGTH:
        raise ValueError(f"{name} must have shape [16, hidden_size]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    actual_hidden = int(value.shape[1])
    if hidden_size is not None and actual_hidden != int(hidden_size):
        raise ValueError(
            f"{name} hidden size {actual_hidden} does not match {hidden_size}"
        )
    return actual_hidden


@dataclass(frozen=True)
class FrozenTransferCheckpoint:
    path: Path
    condition: str
    task: str
    seed: int
    model_name: str
    optimizer_step: int
    sample_order_hash: str
    shared: Any
    task_prefix: Any
    checkpoint_sha256: str
    shared_sha256: str
    task_sha256: str
    source_metadata: dict[str, Any]

    @property
    def hidden_size(self) -> int:
        return int(self.shared.shape[1])

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "condition": self.condition,
            "task": self.task,
            "seed": self.seed,
            "model_name": self.model_name,
            "optimizer_step": self.optimizer_step,
            "sample_order_hash": self.sample_order_hash,
            "checkpoint_sha256": self.checkpoint_sha256,
            "shared": {
                "shape": list(self.shared.shape),
                "dtype": str(self.shared.dtype).removeprefix("torch."),
                "sha256": self.shared_sha256,
            },
            "task_prefix": {
                "shape": list(self.task_prefix.shape),
                "dtype": str(self.task_prefix.dtype).removeprefix("torch."),
                "sha256": self.task_sha256,
            },
            "source_metadata": self.source_metadata or None,
        }


def load_frozen_transfer_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    expected_condition: str,
    expected_task: str,
    expected_seed: int,
    torch_module: Any,
) -> FrozenTransferCheckpoint:
    """Load exactly the two final 16-token blocks from an F0 or F1 checkpoint."""
    expected_condition = str(expected_condition).strip().lower()
    expected_task = str(expected_task).strip().lower()
    if expected_condition not in {"f0", "f1"}:
        raise ValueError("cross composition only accepts F0 and F1 checkpoints")
    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    checkpoint = _torch_load(torch_module, source)
    for key, expected in (
        ("protocol_version", TRAIN_CHECKPOINT_PROTOCOL),
        ("task", expected_task),
        ("condition", expected_condition),
        ("seed", int(expected_seed)),
        ("optimizer_step", FINAL_OPTIMIZER_STEP),
    ):
        if checkpoint.get(key) != expected:
            raise RuntimeError(
                f"{expected_condition.upper()} checkpoint {key} mismatch: "
                f"expected {expected!r}, got {checkpoint.get(key)!r}"
            )
    sample_order_hash = str(checkpoint.get("sample_order_hash", ""))
    if len(sample_order_hash) != 64:
        raise RuntimeError("checkpoint has no valid sample_order_hash")

    state = checkpoint.get("prefix_state")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint has no prefix_state")
    for key, expected in (
        ("protocol_version", "new_task_transfer_v1"),
        ("task_name", expected_task),
        ("condition", expected_condition.upper()),
        ("prefix_length", PREFIX_BLOCK_LENGTH),
        ("effective_prefix_length", EFFECTIVE_PREFIX_LENGTH),
        ("use_residual_reparameterization", False),
    ):
        if state.get(key) != expected:
            raise RuntimeError(
                f"{expected_condition.upper()} prefix_state {key} mismatch"
            )
    model_name = str(state.get("model_name", "")).strip()
    if not model_name:
        raise RuntimeError("checkpoint prefix_state has no model_name")
    tensors = state.get("prefix_tensors")
    if not isinstance(tensors, dict) or set(tensors) != {
        "shared_prefix_embeddings",
        "task_prefix_embeddings",
    }:
        raise RuntimeError("F0/F1 checkpoint must contain exactly Shared16 and Task16")
    shared = tensors["shared_prefix_embeddings"]
    task_prefix = tensors["task_prefix_embeddings"]
    hidden_size = _validate_tensor(
        torch_module, shared, name="shared_prefix_embeddings", hidden_size=None
    )
    _validate_tensor(
        torch_module,
        task_prefix,
        name="task_prefix_embeddings",
        hidden_size=hidden_size,
    )
    if shared.dtype != task_prefix.dtype:
        raise TypeError("Shared16 and Task16 checkpoint dtypes differ")
    shared = shared.detach().cpu().contiguous().clone()
    task_prefix = task_prefix.detach().cpu().contiguous().clone()
    return FrozenTransferCheckpoint(
        path=source,
        condition=expected_condition,
        task=expected_task,
        seed=int(expected_seed),
        model_name=model_name,
        optimizer_step=FINAL_OPTIMIZER_STEP,
        sample_order_hash=sample_order_hash,
        shared=shared,
        task_prefix=task_prefix,
        checkpoint_sha256=file_sha256(source),
        shared_sha256=tensor_sha256(shared),
        task_sha256=tensor_sha256(task_prefix),
        source_metadata=dict(state.get("source_checkpoint") or {}),
    )


def validate_checkpoint_pair(
    f0: FrozenTransferCheckpoint,
    f1: FrozenTransferCheckpoint,
) -> None:
    if f0.condition != "f0" or f1.condition != "f1":
        raise RuntimeError("checkpoint pair must be ordered as F0 then F1")
    for attribute in ("task", "seed", "model_name", "sample_order_hash", "hidden_size"):
        if getattr(f0, attribute) != getattr(f1, attribute):
            raise RuntimeError(f"F0/F1 checkpoint {attribute} mismatch")
    if f0.shared.dtype != f1.shared.dtype:
        raise TypeError("F0/F1 checkpoint dtypes differ")


def compose_prefix_blocks(
    f0: FrozenTransferCheckpoint,
    f1: FrozenTransferCheckpoint,
    composition: str,
) -> tuple[Any, Any, dict[str, Any]]:
    validate_checkpoint_pair(f0, f1)
    composition = str(composition).strip().lower()
    if composition not in COMPOSITION_SOURCES:
        raise ValueError(f"unknown composition: {composition}")
    shared_source, task_source = COMPOSITION_SOURCES[composition]
    checkpoints = {"f0": f0, "f1": f1}
    shared = checkpoints[shared_source].shared.detach().clone()
    task_prefix = checkpoints[task_source].task_prefix.detach().clone()
    active = f0.shared.new_empty((EFFECTIVE_PREFIX_LENGTH, f0.hidden_size))
    active[:PREFIX_BLOCK_LENGTH].copy_(shared)
    active[PREFIX_BLOCK_LENGTH:].copy_(task_prefix)
    return shared, task_prefix, {
        "name": composition,
        "shared_source": shared_source,
        "task_source": task_source,
        "effective_prefix_length": EFFECTIVE_PREFIX_LENGTH,
        "shared_sha256": tensor_sha256(shared),
        "task_sha256": tensor_sha256(task_prefix),
        "active_prefix_sha256": tensor_sha256(active),
    }


def install_composed_prefix(
    model: Any,
    *,
    shared: Any,
    task_prefix: Any,
) -> dict[str, Any]:
    """Copy a validated composition into one F0-shaped evaluation wrapper."""
    if str(getattr(model, "condition", "")).upper() != "F0":
        raise RuntimeError("cross evaluation model must use the F0 parameter layout")
    if bool(getattr(model, "use_residual_reparameterization", True)):
        raise RuntimeError("cross composition requires residual=False")
    if any(parameter.requires_grad for parameter in model.model.parameters()):
        raise RuntimeError("cross composition requires a fully frozen base model")
    destination_shared = model.prefix_parameters.shared_prefix
    destination_task = model.prefix_parameters.task_prefix
    for name, source, destination in (
        ("Shared16", shared, destination_shared),
        ("Task16", task_prefix, destination_task),
    ):
        if tuple(source.shape) != tuple(destination.shape):
            raise ValueError(f"{name} shape does not match the loaded model")
        if source.dtype != destination.dtype:
            raise TypeError(f"{name} dtype does not match the loaded model")
    with model.torch.no_grad():
        destination_shared.copy_(shared.to(device=destination_shared.device))
        destination_task.copy_(task_prefix.to(device=destination_task.device))
    active = model.active_prefix_embeddings()
    if tuple(active.shape) != (EFFECTIVE_PREFIX_LENGTH, int(shared.shape[1])):
        raise RuntimeError("installed cross-composed prefix is not 32 tokens")
    return {
        "effective_prefix_length": int(active.shape[0]),
        "shared_sha256": tensor_sha256(model.shared_prefix_tensor()),
        "task_sha256": tensor_sha256(model.task_prefix_tensor()),
        "active_prefix_sha256": tensor_sha256(active),
        "base_model_fully_frozen": True,
        "residual_enabled": False,
    }


def _cosine(torch_module: Any, left: Any, right: Any) -> float:
    left = left.detach().float().reshape(1, -1)
    right = right.detach().float().reshape(1, -1)
    return float(torch_module.nn.functional.cosine_similarity(left, right).item())


def checkpoint_geometry(
    f0: FrozenTransferCheckpoint,
    f1: FrozenTransferCheckpoint,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    validate_checkpoint_pair(f0, f1)

    def compare(left: Any, right: Any) -> dict[str, float]:
        left_float = left.detach().float()
        right_float = right.detach().float()
        delta = right_float - left_float
        left_norm = float(left_float.norm().item())
        return {
            "f0_norm": left_norm,
            "f1_norm": float(right_float.norm().item()),
            "delta_norm": float(delta.norm().item()),
            "relative_delta_to_f0": float(delta.norm().item()) / left_norm,
            "cosine_similarity": _cosine(torch_module, left_float, right_float),
        }

    return {
        "shared_raw_f0_vs_transferred_f1": compare(f0.shared, f1.shared),
        "final_task_f0_vs_f1": compare(f0.task_prefix, f1.task_prefix),
    }


def deterministic_subset(
    dataset: Iterable[dict[str, Any]],
    *,
    task: str,
    split: str,
    seed: int,
    max_samples: int,
) -> tuple[list[dict[str, Any]], str]:
    items = list(dataset)
    if not items:
        raise ValueError("evaluation dataset cannot be empty")
    if int(max_samples) < 0:
        raise ValueError("max_samples cannot be negative")
    selected = items
    if max_samples and max_samples < len(items):
        ranked = sorted(
            range(len(items)),
            key=lambda index: hashlib.sha256(
                f"{task}:{split}:{int(seed)}:{items[index]['id']}".encode("utf-8")
            ).digest(),
        )[: int(max_samples)]
        selected_indices = set(ranked)
        selected = [item for index, item in enumerate(items) if index in selected_indices]
    digest = hashlib.sha256(
        json.dumps(
            [str(item["id"]) for item in selected],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return selected, digest


def run_cross_composition_evaluation(
    *,
    model: Any,
    f0: FrozenTransferCheckpoint,
    f1: FrozenTransferCheckpoint,
    dataset: Iterable[dict[str, Any]],
    split: str,
    output_dir: str | os.PathLike[str],
    compositions: Iterable[str] = DEFAULT_COMPOSITIONS,
    batch_size: int = 1,
    generation_config: dict[str, Any] | None = None,
    dataset_id_hash: str = "",
    evaluator: Callable[..., dict[str, Any]] = evaluate_transfer_predictions,
) -> dict[str, Any]:
    """Evaluate requested native/crossed prefixes and atomically publish summary."""
    validate_checkpoint_pair(f0, f1)
    requested = tuple(str(value).strip().lower() for value in compositions)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("compositions must be a non-empty unique sequence")
    unknown = sorted(set(requested).difference(COMPOSITION_SOURCES))
    if unknown:
        raise ValueError(f"unknown compositions: {unknown}")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"cross-composition output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "cross_composition_summary.json"
    dataset_items = list(dataset)
    if not dataset_items:
        raise ValueError("evaluation dataset cannot be empty")

    results: dict[str, Any] = {}
    started = time.perf_counter()
    for composition in requested:
        shared, task_prefix, composition_record = compose_prefix_blocks(
            f0, f1, composition
        )
        installed = install_composed_prefix(
            model, shared=shared, task_prefix=task_prefix
        )
        if installed["active_prefix_sha256"] != composition_record["active_prefix_sha256"]:
            raise RuntimeError("installed prefix differs from composed checkpoint tensors")
        predictions_path = output / "predictions" / f"{composition}.jsonl"
        condition_started = time.perf_counter()
        evaluation = evaluator(
            model=model,
            task=f0.task,
            dataset=dataset_items,
            split=split,
            batch_size=batch_size,
            generation_config=generation_config,
            predictions_path=predictions_path,
        )
        if not bool(evaluation.get("complete", False)) or not predictions_path.is_file():
            raise RuntimeError(f"incomplete evaluation for {composition}")
        results[composition] = {
            **composition_record,
            "installed_sanity": installed,
            "metrics": dict(evaluation["metrics"]),
            "num_samples": int(evaluation["num_samples"]),
            "generation": dict(evaluation["generation"]),
            "predictions_path": str(predictions_path),
            "elapsed_seconds": time.perf_counter() - condition_started,
        }

    summary = {
        "status": "complete",
        "protocol": "new_task_transfer_cross_composition_v1",
        "task": f0.task,
        "seed": f0.seed,
        "split": str(split),
        "num_samples": len(dataset_items),
        "dataset_id_hash": str(dataset_id_hash),
        "sample_order_hash": f0.sample_order_hash,
        "checkpoints": {"f0": f0.metadata(), "f1": f1.metadata()},
        "geometry": checkpoint_geometry(f0, f1, torch_module=model.torch),
        "compositions": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_write_json(summary_path, summary)
    return summary
