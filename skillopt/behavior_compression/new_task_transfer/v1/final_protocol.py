"""Stage-3C2 final-only training/evaluation artifact orchestration."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import (
    tensor_sha256,
)
from skillopt.behavior_compression.new_task_transfer.v1.evaluation import (
    _resolve_generation,
    evaluate_transfer_predictions,
)
from skillopt.behavior_compression.new_task_transfer.v1.train_core import (
    MinimalTrainConfig,
)
from skillopt.behavior_compression.new_task_transfer.v1.train_state import (
    run_stateful_training,
)


FINAL_OPTIMIZER_STEPS = 150
EXPECTED_EVAL_SPLITS = {
    "chartqa": ("validation", "test"),
    "drop": ("validation",),
}
REPORT_SPLITS = {"chartqa": "test", "drop": "validation"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL line {line_number} is not an object: {path}")
        records.append(value)
    return records


def _torch_load(torch_module: Any, path: Path) -> dict[str, Any]:
    try:
        value = torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch_module.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint must be a dictionary: {path}")
    return value


def _atomic_torch_save(torch_module: Any, payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch_module.save(payload, temporary)
    os.replace(temporary, path)


class _UniqueJsonlLedger:
    def __init__(self, path: Path, key: str) -> None:
        self.path = path
        self.key = key
        self.records = _read_jsonl(path)
        keys = [record.get(key) for record in self.records]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate {key} records in {path}")

    def get(self, value: Any) -> dict[str, Any] | None:
        return next((record for record in self.records if record.get(self.key) == value), None)

    def append(self, record: dict[str, Any]) -> None:
        value = record.get(self.key)
        if value is None:
            raise RuntimeError(f"record lacks ledger key {self.key}")
        existing = self.get(value)
        if existing is not None:
            if existing != record:
                raise RuntimeError(f"conflicting duplicate {self.key}={value}")
            return
        if self.key == "optimizer_step":
            expected = len(self.records) + 1
            if int(value) != expected:
                raise RuntimeError(
                    f"train metric step must be consecutive: expected {expected}, got {value}"
                )
        self.records.append(dict(record))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self.path, self.records)


@contextmanager
def _preserve_rng_state(model: Any) -> Iterator[None]:
    python_state = random.getstate()
    torch_module = getattr(model, "torch", None)
    torch_state = torch_module.get_rng_state() if torch_module is not None else None
    cuda_states = None
    if (
        torch_module is not None
        and hasattr(torch_module, "cuda")
        and torch_module.cuda.is_available()
    ):
        cuda_states = torch_module.cuda.get_rng_state_all()
    numpy_module = sys.modules.get("numpy")
    numpy_state = (
        numpy_module.random.get_state() if numpy_module is not None else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        if torch_module is not None and torch_state is not None:
            torch_module.set_rng_state(torch_state)
        if cuda_states is not None:
            torch_module.cuda.set_rng_state_all(cuda_states)
        if numpy_module is not None and numpy_state is not None:
            numpy_module.random.set_state(numpy_state)


def _validate_identity(payload: dict[str, Any], config: MinimalTrainConfig, name: str) -> None:
    for key, expected in (
        ("task", config.task),
        ("condition", config.condition),
        ("seed", config.seed),
    ):
        if payload.get(key) != expected:
            raise RuntimeError(f"{name} {key} mismatch")


def _validate_final_checkpoint(
    checkpoint: dict[str, Any],
    *,
    config: MinimalTrainConfig,
    sample_order_hash: str,
) -> None:
    _validate_identity(checkpoint, config, "final checkpoint")
    if int(checkpoint.get("optimizer_step", -1)) != FINAL_OPTIMIZER_STEPS:
        raise RuntimeError("final checkpoint is not optimizer step 150")
    if checkpoint.get("sample_order_hash") != sample_order_hash:
        raise RuntimeError("final checkpoint sample order hash mismatch")
    if not isinstance(checkpoint.get("prefix_state"), dict):
        raise RuntimeError("final checkpoint lacks prefix-only state")


def _reconcile_last_train_metric(
    *,
    model: Any,
    checkpoint_path: Path,
    ledger: _UniqueJsonlLedger,
) -> None:
    if not checkpoint_path.is_file():
        return
    checkpoint = _torch_load(model.torch, checkpoint_path)
    record = checkpoint.get("last_train_metric")
    if record is not None:
        if not isinstance(record, dict):
            raise RuntimeError("checkpoint last_train_metric must be a dictionary")
        ledger.append(record)


def _expected_sample_ids(dataset: Any) -> list[str]:
    items = dataset.items if hasattr(dataset, "items") else dataset
    return [str(item["id"]) for item in items]


def _aggregate_existing_predictions(
    *,
    task: str,
    predictions_path: Path,
    dataset: Any,
) -> tuple[dict[str, float], int]:
    records = _read_jsonl(predictions_path)
    expected_ids = _expected_sample_ids(dataset)
    actual_ids = [str(record.get("sample_id")) for record in records]
    if actual_ids != expected_ids:
        raise RuntimeError("existing predictions are incomplete or out of source order")
    if task == "chartqa":
        scores = [float(record["relaxed_accuracy"]) for record in records]
        return {"relaxed_accuracy": sum(scores) / len(scores)}, len(records)
    em = [float(record["em"]) for record in records]
    f1 = [float(record["f1"]) for record in records]
    return {"em": sum(em) / len(em), "f1": sum(f1) / len(f1)}, len(records)


def _eval_record(
    *,
    config: MinimalTrainConfig,
    split: str,
    evaluation: dict[str, Any],
    elapsed_seconds: float | None,
    recovered: bool,
) -> dict[str, Any]:
    return {
        "task": config.task,
        "condition": config.condition,
        "seed": config.seed,
        "checkpoint": "final.pt",
        "optimizer_step": FINAL_OPTIMIZER_STEPS,
        "split": split,
        "report_split": split == REPORT_SPLITS[config.task],
        "num_samples": int(evaluation["num_samples"]),
        "generation": dict(evaluation["generation"]),
        "metrics": dict(evaluation["metrics"]),
        "elapsed_seconds": elapsed_seconds,
        "recovered_from_complete_predictions": bool(recovered),
    }


def _validate_training_artifacts(
    *,
    config: MinimalTrainConfig,
    output_dir: Path,
    train_ledger: _UniqueJsonlLedger,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    audit = _read_json(output_dir / "step0_audit.json")
    sanity = _read_json(output_dir / "transfer_sanity.json")
    _validate_identity(audit, config, "step0 audit")
    _validate_identity(sanity, config, "transfer sanity")
    if audit.get("status") != "passed" or sanity.get("status") != "passed":
        raise RuntimeError("audit and first-step sanity must both pass")
    if int(audit.get("prefix_total_length", -1)) != 32:
        raise RuntimeError("final-only protocol requires a 32-token prefix")
    if not bool(audit.get("base_model", {}).get("fully_frozen", False)):
        raise RuntimeError("base model is not fully frozen")
    if bool(audit.get("residual_enabled", True)):
        raise RuntimeError("residual must remain disabled")
    steps = [int(record.get("optimizer_step", -1)) for record in train_ledger.records]
    if steps != list(range(1, FINAL_OPTIMIZER_STEPS + 1)):
        raise RuntimeError("train_metrics.jsonl must contain each step 1..150 exactly once")
    return audit, sanity, str(audit["sample_order_hash"])


def _shared_summary(model: Any, config: MinimalTrainConfig, audit: dict[str, Any]) -> dict[str, Any]:
    final_hash = tensor_sha256(model.shared_prefix_tensor())
    if config.condition == "f1":
        source = dict(audit.get("f1_source") or {})
        expected_hash = str(source.get("shared_tensor_sha256", ""))
        if not expected_hash or final_hash != expected_hash:
            raise RuntimeError("F1 Shared16 changed or no longer matches its source")
        return {
            "mode": "transferred_shared16_frozen",
            "final_tensor_sha256": final_hash,
            "source": source,
        }
    initialization = dict(audit.get("initialization_token_info_reference") or {})
    return {
        "mode": (
            "raw_shared16_frozen" if config.condition == "f0" else "target_owned_i32"
        ),
        "final_first16_sha256": final_hash,
        "initialization_reference": initialization.get("shared_behavior"),
    }


def _completed_summary_if_present(
    path: Path,
    *,
    config: MinimalTrainConfig,
    resume: bool,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not resume:
        raise RuntimeError("summary already exists; explicit resume is required")
    summary = _read_json(path)
    _validate_identity(summary, config, "summary")
    if summary.get("status") != "complete":
        raise RuntimeError("only a complete summary may exist in final-only protocol")
    return summary


def run_training_to_final_checkpoint(
    *,
    config: MinimalTrainConfig,
    model: Any,
    train_dataset: Any,
    collator: Any,
    resume: bool = False,
    resume_checkpoint: str = "",
    checkpoint_every_steps: int = 50,
    training_runner: Callable[..., dict[str, Any]] = run_stateful_training,
) -> dict[str, Any]:
    """Train or validate through step 150 and stop before formal evaluation.

    This is used by process-isolated evaluation launchers.  It preserves the
    exact stage-3 training/checkpoint semantics and deliberately does not write
    eval metrics, predictions, or a complete summary.
    """
    config = config.validated()
    if config.max_optimizer_steps != FINAL_OPTIMIZER_STEPS:
        raise ValueError("training-only protocol requires exactly 150 optimizer steps")
    output_dir = Path(config.output_dir).expanduser().resolve()
    train_ledger = _UniqueJsonlLedger(output_dir / "train_metrics.jsonl", "optimizer_step")
    if resume:
        _reconcile_last_train_metric(
            model=model,
            checkpoint_path=output_dir / "latest.pt",
            ledger=train_ledger,
        )

    final_path = output_dir / "final.pt"
    latest_path = output_dir / "latest.pt"
    if resume and not final_path.is_file() and latest_path.is_file():
        audit_for_hash = _read_json(output_dir / "step0_audit.json")
        latest_checkpoint = _torch_load(model.torch, latest_path)
        if int(latest_checkpoint.get("optimizer_step", -1)) == FINAL_OPTIMIZER_STEPS:
            _validate_final_checkpoint(
                latest_checkpoint,
                config=config,
                sample_order_hash=str(audit_for_hash["sample_order_hash"]),
            )
            _atomic_torch_save(model.torch, latest_checkpoint, final_path)
    if resume and final_path.is_file():
        audit_for_hash = _read_json(output_dir / "step0_audit.json")
        sample_order_hash = str(audit_for_hash["sample_order_hash"])
        final_checkpoint = _torch_load(model.torch, final_path)
        _validate_final_checkpoint(
            final_checkpoint,
            config=config,
            sample_order_hash=sample_order_hash,
        )
        model.load_state_dict(final_checkpoint["prefix_state"])
        _reconcile_last_train_metric(
            model=model,
            checkpoint_path=final_path,
            ledger=train_ledger,
        )
    else:
        training_runner(
            config=config,
            model=model,
            train_dataset=train_dataset,
            collator=collator,
            resume=resume,
            resume_checkpoint=resume_checkpoint,
            checkpoint_every_steps=checkpoint_every_steps,
            optimizer_step_callback=train_ledger.append,
        )
        if not final_path.is_file():
            raise RuntimeError("training returned without final.pt")

    audit, sanity, sample_order_hash = _validate_training_artifacts(
        config=config,
        output_dir=output_dir,
        train_ledger=train_ledger,
    )
    final_checkpoint = _torch_load(model.torch, final_path)
    _validate_final_checkpoint(
        final_checkpoint,
        config=config,
        sample_order_hash=sample_order_hash,
    )
    model.load_state_dict(final_checkpoint["prefix_state"])
    return {
        "status": "training_complete",
        "task": config.task,
        "condition": config.condition,
        "seed": config.seed,
        "optimizer_steps": FINAL_OPTIMIZER_STEPS,
        "sample_order_hash": sample_order_hash,
        "checkpoint_path": str(final_path),
        "prefix_total_length": int(audit["prefix_total_length"]),
        "base_model_fully_frozen": bool(audit["base_model"]["fully_frozen"]),
        "residual_enabled": bool(audit["residual_enabled"]),
        "sanity_status": str(sanity["status"]),
    }


def run_final_only_experiment(
    *,
    config: MinimalTrainConfig,
    model: Any,
    train_dataset: Any,
    collator: Any,
    evaluation_splits: dict[str, Any],
    eval_batch_size: int = 1,
    generation_config: dict[str, Any] | None = None,
    resume: bool = False,
    resume_checkpoint: str = "",
    checkpoint_every_steps: int = 50,
    training_runner: Callable[..., dict[str, Any]] = run_stateful_training,
    evaluator: Callable[..., dict[str, Any]] = evaluate_transfer_predictions,
) -> dict[str, Any]:
    """Train to step 150, evaluate final.pt only, then publish complete summary."""
    config = config.validated()
    if config.max_optimizer_steps != FINAL_OPTIMIZER_STEPS:
        raise ValueError("final-only protocol requires exactly 150 optimizer steps")
    output_dir = Path(config.output_dir).expanduser().resolve()
    summary_path = output_dir / "summary.json"
    completed = _completed_summary_if_present(
        summary_path,
        config=config,
        resume=resume,
    )
    if completed is not None:
        return completed

    expected_splits = EXPECTED_EVAL_SPLITS[config.task]
    if any(split not in evaluation_splits for split in expected_splits):
        raise ValueError(f"evaluation_splits must contain {expected_splits}")
    train_ledger = _UniqueJsonlLedger(output_dir / "train_metrics.jsonl", "optimizer_step")
    if resume:
        _reconcile_last_train_metric(
            model=model,
            checkpoint_path=output_dir / "latest.pt",
            ledger=train_ledger,
        )

    final_path = output_dir / "final.pt"
    latest_path = output_dir / "latest.pt"
    if resume and not final_path.is_file() and latest_path.is_file():
        audit_for_hash = _read_json(output_dir / "step0_audit.json")
        latest_checkpoint = _torch_load(model.torch, latest_path)
        if int(latest_checkpoint.get("optimizer_step", -1)) == FINAL_OPTIMIZER_STEPS:
            _validate_final_checkpoint(
                latest_checkpoint,
                config=config,
                sample_order_hash=str(audit_for_hash["sample_order_hash"]),
            )
            _atomic_torch_save(model.torch, latest_checkpoint, final_path)
    if resume and final_path.is_file():
        audit_for_hash = _read_json(output_dir / "step0_audit.json")
        sample_order_hash = str(audit_for_hash["sample_order_hash"])
        final_checkpoint = _torch_load(model.torch, final_path)
        _validate_final_checkpoint(
            final_checkpoint,
            config=config,
            sample_order_hash=sample_order_hash,
        )
        model.load_state_dict(final_checkpoint["prefix_state"])
        _reconcile_last_train_metric(
            model=model,
            checkpoint_path=final_path,
            ledger=train_ledger,
        )
    else:
        training_runner(
            config=config,
            model=model,
            train_dataset=train_dataset,
            collator=collator,
            resume=resume,
            resume_checkpoint=resume_checkpoint,
            checkpoint_every_steps=checkpoint_every_steps,
            optimizer_step_callback=train_ledger.append,
        )
        if not final_path.is_file():
            raise RuntimeError("training returned without final.pt")

    audit, sanity, sample_order_hash = _validate_training_artifacts(
        config=config,
        output_dir=output_dir,
        train_ledger=train_ledger,
    )
    final_checkpoint = _torch_load(model.torch, final_path)
    _validate_final_checkpoint(
        final_checkpoint,
        config=config,
        sample_order_hash=sample_order_hash,
    )
    # Evaluation always consumes the serialized step-150 prefix, including on
    # a fresh run, rather than relying on an equivalent in-memory assumption.
    model.load_state_dict(final_checkpoint["prefix_state"])

    eval_ledger = _UniqueJsonlLedger(output_dir / "eval_metrics.jsonl", "split")
    predictions_dir = output_dir / "predictions"
    resolved_generation = _resolve_generation(config.task, generation_config)
    for split in expected_splits:
        predictions_path = predictions_dir / f"{split}.jsonl"
        existing_metric = eval_ledger.get(split)
        if existing_metric is not None:
            if not predictions_path.is_file():
                raise RuntimeError(f"eval metric exists without predictions for {split}")
            _aggregate_existing_predictions(
                task=config.task,
                predictions_path=predictions_path,
                dataset=evaluation_splits[split],
            )
            continue
        if predictions_path.is_file():
            scores, sample_count = _aggregate_existing_predictions(
                task=config.task,
                predictions_path=predictions_path,
                dataset=evaluation_splits[split],
            )
            evaluation = {
                "num_samples": sample_count,
                "generation": {
                    **resolved_generation,
                    "batch_size": int(eval_batch_size),
                    "message_order": (
                        "question_then_image"
                        if config.task == "chartqa"
                        else "passage_then_question"
                    ),
                },
                "metrics": scores,
            }
            eval_ledger.append(
                _eval_record(
                    config=config,
                    split=split,
                    evaluation=evaluation,
                    elapsed_seconds=None,
                    recovered=True,
                )
            )
            continue

        evaluation_started = time.perf_counter()
        with _preserve_rng_state(model):
            evaluation = evaluator(
                model=model,
                task=config.task,
                dataset=evaluation_splits[split],
                split=split,
                batch_size=eval_batch_size,
                generation_config=generation_config,
                predictions_path=predictions_path,
                resume_partial=True,
                progress_path=predictions_dir / f"{split}.progress.jsonl",
            )
        if not bool(evaluation.get("complete", False)) or not predictions_path.is_file():
            raise RuntimeError(f"{split} evaluation did not publish complete predictions")
        eval_ledger.append(
            _eval_record(
                config=config,
                split=split,
                evaluation=evaluation,
                elapsed_seconds=time.perf_counter() - evaluation_started,
                recovered=False,
            )
        )

    if {str(record["split"]) for record in eval_ledger.records} != set(expected_splits):
        raise RuntimeError("eval_metrics.jsonl does not contain exactly the formal splits")
    eval_by_split = {str(record["split"]): record for record in eval_ledger.records}
    for split in expected_splits:
        predictions_path = predictions_dir / f"{split}.jsonl"
        _aggregate_existing_predictions(
            task=config.task,
            predictions_path=predictions_path,
            dataset=evaluation_splits[split],
        )

    report_split = REPORT_SPLITS[config.task]
    training_seconds = sum(
        float(record.get("step_elapsed_seconds", 0.0)) for record in train_ledger.records
    )
    known_eval_seconds = [
        float(record["elapsed_seconds"])
        for record in eval_ledger.records
        if record.get("elapsed_seconds") is not None
    ]
    evaluation_seconds = sum(known_eval_seconds)
    summary = {
        "status": "complete",
        "protocol": "new_task_transfer_v1_final_only",
        "task": config.task,
        "condition": config.condition,
        "seed": config.seed,
        "optimizer_steps": FINAL_OPTIMIZER_STEPS,
        "checkpoint": {
            "path": str(final_path),
            "optimizer_step": FINAL_OPTIMIZER_STEPS,
            "selection": "final_only_no_validation_best",
        },
        "report_split": report_split,
        "metrics": dict(eval_by_split[report_split]["metrics"]),
        "evaluation_by_split": {
            split: dict(eval_by_split[split]["metrics"]) for split in expected_splits
        },
        "sample_order_hash": sample_order_hash,
        "shared": _shared_summary(model, config, audit),
        "trainable_prefix_tokens": 32 if config.condition == "i32" else 16,
        "prefix_total_length": 32,
        "sanity": {
            "status": sanity["status"],
            "all_checks_passed": all(
                bool(check.get("passed", False))
                for check in sanity.get("checks", {}).values()
            ),
        },
        "base_model_fully_frozen": bool(audit["base_model"]["fully_frozen"]),
        "residual_enabled": False,
        "timing_seconds": {
            "training_optimizer_steps": training_seconds,
            "evaluation_known": evaluation_seconds,
            "total_known": training_seconds + evaluation_seconds,
            "all_evaluation_times_known": len(known_eval_seconds) == len(expected_splits),
        },
        "artifacts": {
            "train_metrics": str(output_dir / "train_metrics.jsonl"),
            "eval_metrics": str(output_dir / "eval_metrics.jsonl"),
            "predictions": {
                split: str(predictions_dir / f"{split}.jsonl")
                for split in expected_splits
            },
        },
    }
    _write_json(summary_path, summary)
    return summary
