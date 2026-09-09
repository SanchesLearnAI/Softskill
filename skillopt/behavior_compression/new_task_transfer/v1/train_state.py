"""Stage-3B audit, first-step sanity, and optimizer-boundary resume support."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import (
    tensor_sha256,
)
from skillopt.behavior_compression.new_task_transfer.v1.train_core import (
    MinimalTrainConfig,
    _batch_to_device,
    build_deterministic_batch_schedule,
    supervised_token_count,
)


PROTOCOL_VERSION = "new_task_transfer_stage3b_v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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
        raise ValueError(f"checkpoint must be a dictionary: {path}")
    return value


def _atomic_torch_save(torch_module: Any, payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch_module.save(payload, temporary)
    os.replace(temporary, path)


def _resolved_config(
    config: MinimalTrainConfig,
    *,
    checkpoint_every_steps: int,
) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(Path(config.output_dir).expanduser().resolve())
    if config.source_checkpoint:
        payload["source_checkpoint"] = str(
            Path(config.source_checkpoint).expanduser().resolve()
        )
    payload.update(
        {
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "lr_schedule": "constant",
            "checkpoint_every_steps": int(checkpoint_every_steps),
        }
    )
    return payload


def _prepare_output_dir(output_dir: Path, *, resume: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"output_dir is not a directory: {output_dir}")
    contents = list(output_dir.iterdir()) if output_dir.exists() else []
    if resume:
        if not contents:
            raise RuntimeError("resume requires an existing non-empty output directory")
    elif contents:
        raise RuntimeError(
            "output directory is non-empty; pass explicit resume to continue"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _prefix_records(model: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trainable: list[dict[str, Any]] = []
    frozen: list[dict[str, Any]] = []
    for name, parameter in model.prefix_parameters.named_parameters():
        record = {
            "name": name,
            "shape": [int(size) for size in parameter.shape],
            "parameter_count": int(parameter.numel()),
        }
        (trainable if parameter.requires_grad else frozen).append(record)
    return trainable, frozen


def _source_reference(config: MinimalTrainConfig, model: Any) -> dict[str, Any] | None:
    if config.condition != "f1":
        return None
    metadata = dict(getattr(model, "source_metadata", {}) or {})
    required = {
        "checkpoint_path",
        "checkpoint_sha256",
        "state_key",
        "tensor_sha256",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise RuntimeError(f"F1 source metadata is incomplete: {missing}")
    configured_path = str(Path(config.source_checkpoint).expanduser().resolve())
    recorded_path = str(Path(metadata["checkpoint_path"]).expanduser().resolve())
    if configured_path != recorded_path:
        raise RuntimeError("F1 model source does not match explicit source_checkpoint")
    actual_shared_hash = tensor_sha256(model.shared_prefix_tensor())
    if actual_shared_hash != str(metadata["tensor_sha256"]):
        raise RuntimeError("F1 Shared16 does not match the extracted source tensor hash")
    return {
        "checkpoint_path": recorded_path,
        "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
        "state_key": str(metadata["state_key"]),
        "shared_tensor_sha256": str(metadata["tensor_sha256"]),
    }


def build_step0_audit(
    *,
    config: MinimalTrainConfig,
    model: Any,
    sample_order_hash: str,
    checkpoint_every_steps: int,
) -> dict[str, Any]:
    """Validate the condition boundary and build the persisted preflight audit."""
    prefix_length = int(model.active_prefix_embeddings().shape[0])
    base_parameters = list(model.model.parameters())
    base_fully_frozen = not any(parameter.requires_grad for parameter in base_parameters)
    residual_enabled = bool(
        getattr(model, "use_residual_reparameterization", False)
    )
    configured_condition = str(config.condition).lower()
    actual_condition = str(model.condition).lower()
    actual_task = str(model.task_name).lower()
    trainable, frozen = _prefix_records(model)
    trainable_names = {record["name"] for record in trainable}
    frozen_names = {record["name"] for record in frozen}
    expected_trainable = (
        {"independent_prefix"} if configured_condition == "i32" else {"task_prefix"}
    )
    expected_frozen = set() if configured_condition == "i32" else {"shared_prefix"}

    failures: list[str] = []
    if prefix_length != 32:
        failures.append("effective prefix length is not 32")
    if not base_fully_frozen:
        failures.append("base model has trainable parameters")
    if residual_enabled:
        failures.append("residual is enabled")
    if actual_condition != configured_condition:
        failures.append("configured and model conditions differ")
    if actual_task != config.task:
        failures.append("configured and model tasks differ")
    if trainable_names != expected_trainable or frozen_names != expected_frozen:
        failures.append("condition prefix trainable/frozen boundary is incorrect")

    source = None
    try:
        source = _source_reference(config, model)
    except Exception as exc:
        failures.append(str(exc))
    initialization_reference = None
    if configured_condition in {"f0", "i32"}:
        initialization_reference = dict(
            getattr(model, "initialization_audit", {}) or {}
        )

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed" if not failures else "failed",
        "task": config.task,
        "condition": config.condition,
        "seed": config.seed,
        "prefix_total_length": prefix_length,
        "prefix_parameters": {
            "trainable": trainable,
            "frozen": frozen,
            "trainable_parameter_count": sum(
                record["parameter_count"] for record in trainable
            ),
            "frozen_parameter_count": sum(
                record["parameter_count"] for record in frozen
            ),
        },
        "base_model": {
            "fully_frozen": base_fully_frozen,
            "parameter_count": sum(parameter.numel() for parameter in base_parameters),
        },
        "residual_enabled": residual_enabled,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": 0.0,
            "lr_schedule": "constant",
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "target_optimizer_steps": config.max_optimizer_steps,
        },
        "sample_order_hash": sample_order_hash,
        "f1_source": source,
        "initialization_token_info_reference": initialization_reference,
        "failures": failures,
    }
    if failures:
        raise RuntimeError("step0 audit failed: " + "; ".join(failures))
    return audit


def _check(passed: bool, evidence: str) -> dict[str, Any]:
    return {"passed": bool(passed), "evidence": str(evidence)}


def _named_prefix_parameters(model: Any) -> dict[str, Any]:
    return dict(model.prefix_parameters.named_parameters())


def _snapshot_prefixes(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.prefix_parameters.named_parameters()
    }


def build_transfer_sanity(
    *,
    config: MinimalTrainConfig,
    model: Any,
    optimizer: Any,
    prefix_before: dict[str, Any],
    optimizer_step: int,
    source_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate first-step invariants without copying frozen base weights."""
    torch_module = model.torch
    named_prefix = _named_prefix_parameters(model)
    trainable_names = [name for name, value in named_prefix.items() if value.requires_grad]
    frozen_names = [name for name, value in named_prefix.items() if not value.requires_grad]
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    base_parameters = list(model.model.parameters())

    grad_evidence: list[str] = []
    trainable_grads_ok = True
    for name in trainable_names:
        gradient = named_prefix[name].grad
        finite = gradient is not None and bool(torch_module.isfinite(gradient).all())
        nonzero = finite and bool(gradient.detach().abs().max().item() > 0)
        trainable_grads_ok = trainable_grads_ok and finite and nonzero
        grad_evidence.append(f"{name}:finite={finite},nonzero={nonzero}")

    changed = {
        name: not torch_module.equal(prefix_before[name], named_prefix[name].detach())
        for name in trainable_names
    }
    frozen_grad_absent = {
        name: named_prefix[name].grad is None for name in frozen_names
    }
    frozen_unchanged = {
        name: torch_module.equal(prefix_before[name], named_prefix[name].detach())
        for name in frozen_names
    }
    source_matches = True
    source_evidence = "not applicable"
    if config.condition == "f1":
        expected_hash = str((source_reference or {}).get("shared_tensor_sha256", ""))
        before_hash = tensor_sha256(prefix_before["shared_prefix"])
        after_hash = tensor_sha256(named_prefix["shared_prefix"])
        source_matches = bool(expected_hash) and before_hash == expected_hash == after_hash
        source_evidence = (
            f"expected={expected_hash},before={before_hash},after={after_hash}"
        )

    checks = {
        "base_requires_grad_false": _check(
            not any(parameter.requires_grad for parameter in base_parameters),
            f"base_parameter_count={sum(parameter.numel() for parameter in base_parameters)}",
        ),
        "base_gradients_absent": _check(
            all(parameter.grad is None for parameter in base_parameters),
            "all frozen base gradients are None",
        ),
        "base_not_in_optimizer": _check(
            not any(id(parameter) in optimizer_ids for parameter in base_parameters),
            f"optimizer_parameter_tensors={len(optimizer_ids)}",
        ),
        "trainable_prefix_gradients_valid": _check(
            bool(trainable_names) and trainable_grads_ok,
            ",".join(grad_evidence),
        ),
        "trainable_prefix_changed": _check(
            bool(changed) and all(changed.values()),
            json.dumps(changed, sort_keys=True),
        ),
        "frozen_prefix_gradients_absent": _check(
            all(frozen_grad_absent.values()),
            json.dumps(frozen_grad_absent, sort_keys=True),
        ),
        "frozen_prefix_unchanged": _check(
            all(frozen_unchanged.values()),
            json.dumps(frozen_unchanged, sort_keys=True),
        ),
        "f1_shared_matches_source_and_unchanged": _check(
            source_matches,
            source_evidence,
        ),
        "optimizer_step_is_one": _check(
            int(optimizer_step) == 1,
            f"optimizer_step={optimizer_step}",
        ),
        "effective_prefix_length_is_32": _check(
            int(model.active_prefix_embeddings().shape[0]) == 32,
            f"prefix_length={int(model.active_prefix_embeddings().shape[0])}",
        ),
    }
    passed = all(record["passed"] for record in checks.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed" if passed else "failed",
        "task": config.task,
        "condition": config.condition,
        "seed": config.seed,
        "optimizer_step": int(optimizer_step),
        "checks": checks,
    }


def _checkpoint_payload(
    *,
    config: MinimalTrainConfig,
    model: Any,
    optimizer: Any,
    optimizer_step: int,
    microbatches_consumed: int,
    resolved_config: dict[str, Any],
    sample_order_hash: str,
    source_reference: dict[str, Any] | None,
    last_train_metric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": config.task,
        "condition": config.condition,
        "seed": config.seed,
        "optimizer_step": int(optimizer_step),
        "microbatches_consumed": int(microbatches_consumed),
        "prefix_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "resolved_config": resolved_config,
        "sample_order_hash": sample_order_hash,
        "f1_source": source_reference,
        "last_train_metric": last_train_metric,
    }


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    config: MinimalTrainConfig,
    resolved_config: dict[str, Any],
    sample_order_hash: str,
    source_reference: dict[str, Any] | None,
) -> int:
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("resume checkpoint protocol mismatch")
    for key, expected in (
        ("task", config.task),
        ("condition", config.condition),
        ("seed", config.seed),
    ):
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"resume checkpoint {key} mismatch")
    if checkpoint.get("resolved_config") != resolved_config:
        raise RuntimeError("resume resolved config mismatch")
    if checkpoint.get("sample_order_hash") != sample_order_hash:
        raise RuntimeError("resume sample order hash mismatch")
    if checkpoint.get("f1_source") != source_reference:
        raise RuntimeError("resume F1 source path/hash mismatch")
    step = int(checkpoint.get("optimizer_step", -1))
    if step < 0 or step > config.max_optimizer_steps:
        raise RuntimeError("resume optimizer step is outside configured total steps")
    expected_microbatches = step * config.gradient_accumulation_steps
    if int(checkpoint.get("microbatches_consumed", -1)) != expected_microbatches:
        raise RuntimeError("resume microbatch position is inconsistent with optimizer step")
    if not isinstance(checkpoint.get("prefix_state"), dict):
        raise RuntimeError("resume checkpoint lacks prefix state")
    if not isinstance(checkpoint.get("optimizer_state"), dict):
        raise RuntimeError("resume checkpoint lacks optimizer state")
    return step


def run_stateful_training(
    *,
    config: MinimalTrainConfig,
    model: Any,
    train_dataset: Any,
    collator: Any,
    resume: bool = False,
    resume_checkpoint: str = "",
    checkpoint_every_steps: int = 0,
    first_step_post_optimizer_hook: Callable[[Any], None] | None = None,
    after_checkpoint_hook: Callable[[int], None] | None = None,
    optimizer_step_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the 3A protocol with 3B persistence at optimizer-step boundaries."""
    config = config.validated()
    if not str(config.output_dir).strip():
        raise ValueError("output_dir is required for stage 3B")
    if int(checkpoint_every_steps) < 0:
        raise ValueError("checkpoint_every_steps cannot be negative")
    output_dir = Path(config.output_dir).expanduser().resolve()
    _prepare_output_dir(output_dir, resume=resume)

    if callable(getattr(model, "validate_invariants", None)):
        model.validate_invariants()
    trainable = list(model.trainable_parameters())
    if not trainable:
        raise RuntimeError("no trainable target prefix parameters")
    allowed_ids = {
        id(parameter)
        for parameter in model.prefix_parameters.parameters()
        if parameter.requires_grad
    }
    if {id(parameter) for parameter in trainable} != allowed_ids:
        raise RuntimeError("optimizer boundary is not exactly the trainable target prefix")

    sample_ids = [str(item["id"]) for item in train_dataset.items]
    schedule = build_deterministic_batch_schedule(
        sample_ids,
        task=config.task,
        seed=config.seed,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_optimizer_steps=config.max_optimizer_steps,
    )
    resolved = _resolved_config(
        config,
        checkpoint_every_steps=int(checkpoint_every_steps),
    )
    source_reference = _source_reference(config, model)
    audit = build_step0_audit(
        config=config,
        model=model,
        sample_order_hash=schedule.sample_order_hash,
        checkpoint_every_steps=int(checkpoint_every_steps),
    )

    resolved_path = output_dir / "resolved_config.json"
    audit_path = output_dir / "step0_audit.json"
    if resume:
        if not resolved_path.is_file() or not audit_path.is_file():
            raise RuntimeError("resume requires resolved_config.json and step0_audit.json")
        existing_resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing_resolved != resolved:
            raise RuntimeError("resume resolved config file mismatch")
        existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing_audit.get("sample_order_hash") != schedule.sample_order_hash:
            raise RuntimeError("resume step0 audit sample order hash mismatch")
    else:
        _write_json(resolved_path, resolved)
        _write_json(audit_path, audit)

    torch_module = model.torch
    optimizer = torch_module.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    start_step = 0
    microbatches_consumed = 0
    last_train_metric: dict[str, Any] | None = None
    latest_path = output_dir / "latest.pt"
    if resume:
        checkpoint_path = (
            Path(resume_checkpoint).expanduser().resolve()
            if str(resume_checkpoint).strip()
            else latest_path
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = _torch_load(torch_module, checkpoint_path)
        start_step = _validate_resume_checkpoint(
            checkpoint,
            config=config,
            resolved_config=resolved,
            sample_order_hash=schedule.sample_order_hash,
            source_reference=source_reference,
        )
        model.load_state_dict(checkpoint["prefix_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        microbatches_consumed = int(checkpoint["microbatches_consumed"])
        last_train_metric = checkpoint.get("last_train_metric")
        if start_step >= config.max_optimizer_steps:
            raise RuntimeError("resume checkpoint already reached the configured total steps")

    losses: list[float] = []
    model.model.eval()
    sanity_path = output_dir / "transfer_sanity.json"
    for step_index in range(start_step, config.max_optimizer_steps):
        step_started = time.perf_counter()
        prefix_before = _snapshot_prefixes(model) if step_index == 0 else {}
        microbatch_start = step_index * config.gradient_accumulation_steps
        index_batches = schedule.index_batches[
            microbatch_start : microbatch_start + config.gradient_accumulation_steps
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
        optimizer_step = step_index + 1
        losses.append(weighted_loss)
        last_train_metric = {
            "optimizer_step": optimizer_step,
            "loss": weighted_loss,
            "microbatches_consumed": microbatches_consumed,
            "step_elapsed_seconds": time.perf_counter() - step_started,
        }

        if optimizer_step == 1:
            if first_step_post_optimizer_hook is not None:
                first_step_post_optimizer_hook(model)
            sanity = build_transfer_sanity(
                config=config,
                model=model,
                optimizer=optimizer,
                prefix_before=prefix_before,
                optimizer_step=optimizer_step,
                source_reference=source_reference,
            )
            _write_json(sanity_path, sanity)
            if sanity["status"] != "passed":
                raise RuntimeError("first optimizer-step transfer sanity failed")

        checkpoint_payload = _checkpoint_payload(
            config=config,
            model=model,
            optimizer=optimizer,
            optimizer_step=optimizer_step,
            microbatches_consumed=microbatches_consumed,
            resolved_config=resolved,
            sample_order_hash=schedule.sample_order_hash,
            source_reference=source_reference,
            last_train_metric=last_train_metric,
        )
        _atomic_torch_save(torch_module, checkpoint_payload, latest_path)
        if checkpoint_every_steps and optimizer_step % checkpoint_every_steps == 0:
            _atomic_torch_save(
                torch_module,
                checkpoint_payload,
                output_dir / f"checkpoint_step{optimizer_step:06d}.pt",
            )
        if optimizer_step_callback is not None:
            optimizer_step_callback(dict(last_train_metric))
        if after_checkpoint_hook is not None:
            after_checkpoint_hook(optimizer_step)

    if microbatches_consumed != (
        config.max_optimizer_steps * config.gradient_accumulation_steps
    ):
        raise RuntimeError("final microbatch count is inconsistent with total steps")
    final_checkpoint = _checkpoint_payload(
        config=config,
        model=model,
        optimizer=optimizer,
        optimizer_step=config.max_optimizer_steps,
        microbatches_consumed=microbatches_consumed,
        resolved_config=resolved,
        sample_order_hash=schedule.sample_order_hash,
        source_reference=source_reference,
        last_train_metric=last_train_metric,
    )
    _atomic_torch_save(torch_module, final_checkpoint, output_dir / "final.pt")
    return {
        "task": config.task,
        "condition": config.condition,
        "seed": config.seed,
        "start_optimizer_step": start_step,
        "optimizer_steps": config.max_optimizer_steps,
        "microbatches_consumed": microbatches_consumed,
        "sample_order_hash": schedule.sample_order_hash,
        "losses_this_process": losses,
    }
