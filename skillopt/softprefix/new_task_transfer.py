"""Single-target Shared Prefix transfer experiments for ChartQA and DROP.

This module is deliberately separate from the three-task v2 trainer.  It
loads only the learned Shared16 tensor from a v2 checkpoint and never restores
the old task set or any base-model weights.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from skillopt.softprefix.model import SoftPrefixVisionLM, _flatten_prefix_embeddings


PROTOCOL_VERSION = "new_task_transfer_v1"
CONDITION_F0 = "F0"
CONDITION_F1 = "F1"
CONDITION_I32 = "I32"
VALID_CONDITIONS = {CONDITION_F0, CONDITION_F1, CONDITION_I32}
VALID_TASKS = {"chartqa", "drop"}
PREFIX_BLOCK_LENGTH = 16
EFFECTIVE_PREFIX_LENGTH = 32
OPTIMIZER_STEPS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0


def tensor_sha256(tensor: Any) -> str:
    payload = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_behavior_tokens(
    tokenizer: Any,
    text: str,
    *,
    expected_length: int = PREFIX_BLOCK_LENGTH,
) -> dict[str, Any]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    token_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    if len(token_ids) != int(expected_length):
        raise ValueError(
            f"behavior initialization must contain exactly {expected_length} tokens; "
            f"tokenizer produced {len(token_ids)}"
        )
    return {
        "raw_text": text,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "selected_token_ids": token_ids[: int(expected_length)],
        "selected_token_count": int(expected_length),
        "selected_text": tokenizer.decode(token_ids[: int(expected_length)]),
    }


def embed_behavior_block(model: SoftPrefixVisionLM, token_audit: dict[str, Any]):
    torch = model.torch
    token_ids = torch.tensor(
        [token_audit["selected_token_ids"]],
        dtype=torch.long,
        device=model.device,
    )
    with torch.no_grad():
        return (
            model.model.get_input_embeddings()(token_ids)[0]
            .to(dtype=model.prefix_embeddings.dtype)
            .detach()
            .clone()
        )


def compose_transfer_blocks(
    raw_shared_block: Any,
    task_block: Any,
    *,
    condition: str,
    transferred_shared_block: Any | None = None,
) -> tuple[Any, Any]:
    """Return cloned Shared16/Task16 initializations for one condition."""
    condition = str(condition).upper()
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(VALID_CONDITIONS)}")
    if tuple(raw_shared_block.shape) != tuple(task_block.shape):
        raise ValueError("Shared and Task behavior blocks must have the same shape")
    shared = raw_shared_block
    if condition == CONDITION_F1:
        if transferred_shared_block is None:
            raise ValueError("F1 requires a transferred Shared16 block")
        if tuple(transferred_shared_block.shape) != tuple(raw_shared_block.shape):
            raise ValueError("transferred Shared16 shape does not match raw Shared16")
        shared = transferred_shared_block
    elif transferred_shared_block is not None:
        raise ValueError("only F1 accepts a transferred Shared16 block")
    return shared.detach().clone(), task_block.detach().clone()


def load_transferred_shared_prefix(
    checkpoint_path: str | os.PathLike[str],
    *,
    torch_module: Any | None = None,
    prefix_length: int = PREFIX_BLOCK_LENGTH,
) -> tuple[Any, dict[str, Any]]:
    """Extract only ``model.shared_prefix_embeddings`` from a v2 checkpoint."""
    if torch_module is None:
        import torch as torch_module

    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source behavior checkpoint is missing: {source}")
    checkpoint = torch_module.load(source, map_location="cpu")
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("source checkpoint has no model state dictionary")
    if str(state.get("protocol_version")) != "v2":
        raise ValueError("source checkpoint is not a v2 checkpoint")
    if str(state.get("layout")) != "shared16_task16":
        raise ValueError("source checkpoint does not use shared16_task16")
    if str(state.get("initialization_mode")) != "behavior_markdown":
        raise ValueError("source checkpoint does not use behavior_markdown initialization")
    if int(state.get("prefix_length", -1)) != int(prefix_length):
        raise ValueError("source checkpoint Shared prefix length is not 16")
    if bool(state.get("use_residual_reparameterization", False)):
        raise ValueError("source checkpoint must have residual reparameterization disabled")
    value = state.get("shared_prefix_embeddings")
    if value is None:
        raise ValueError("source checkpoint has no model.shared_prefix_embeddings")
    if value.dim() == 3 and int(value.shape[0]) == 1:
        value = value[0]
    if value.dim() != 2 or int(value.shape[0]) != int(prefix_length):
        raise ValueError(f"invalid source Shared prefix shape: {tuple(value.shape)}")
    value = value.detach().cpu().clone()
    metadata = {
        "checkpoint_path": str(source),
        "checkpoint_sha256": file_sha256(source),
        "state_key": "model.shared_prefix_embeddings",
        "shape": list(value.shape),
        "tensor_sha256": tensor_sha256(value),
        "source_protocol_version": str(state.get("protocol_version")),
        "source_layout": str(state.get("layout")),
        "source_initialization_mode": str(state.get("initialization_mode", "")),
        "source_residual_enabled": bool(
            state.get("use_residual_reparameterization", False)
        ),
    }
    return value, metadata


class NewTaskTransferSoftPrefixVisionLM(SoftPrefixVisionLM):
    """Frozen Qwen with F0/F1 frozen Shared16 or trainable independent I32."""

    def __init__(
        self,
        model_name: str,
        *,
        condition: str,
        shared_behavior_text: str,
        task_behavior_text: str,
        source_checkpoint_path: str = "",
        torch_dtype: str = "auto",
        device: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        condition = str(condition).upper()
        if condition not in VALID_CONDITIONS:
            raise ValueError(f"condition must be one of {sorted(VALID_CONDITIONS)}")
        super().__init__(
            model_name,
            prefix_length=PREFIX_BLOCK_LENGTH,
            num_soft_skills=2 if condition == CONDITION_I32 else 1,
            init_text="",
            init_strategy="random",
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        self.protocol_version = PROTOCOL_VERSION
        self.condition = condition
        self.use_residual_reparameterization = False
        if condition == CONDITION_F1 and not str(source_checkpoint_path).strip():
            raise ValueError("F1 requires source_checkpoint_path")
        if condition != CONDITION_F1 and str(source_checkpoint_path).strip():
            raise ValueError("only F1 accepts source_checkpoint_path")
        self.source_metadata: dict[str, Any] = {}
        shared_audit = audit_behavior_tokens(self.tokenizer, shared_behavior_text)
        task_audit = audit_behavior_tokens(self.tokenizer, task_behavior_text)
        raw_shared = embed_behavior_block(self, shared_audit)
        task = embed_behavior_block(self, task_audit)
        transferred = None
        if condition == CONDITION_F1:
            transferred_cpu, self.source_metadata = load_transferred_shared_prefix(
                source_checkpoint_path,
                torch_module=self.torch,
            )
            transferred = transferred_cpu.to(
                device=self.device,
                dtype=self.prefix_embeddings.dtype,
            )
        shared, task = compose_transfer_blocks(
            raw_shared,
            task,
            condition=condition,
            transferred_shared_block=transferred,
        )

        self.task_prefix_embeddings = None
        with self.torch.no_grad():
            if condition == CONDITION_I32:
                self.prefix_embeddings.copy_(self.torch.stack([shared, task], dim=0))
            else:
                self.prefix_embeddings.copy_(shared.unsqueeze(0))
                self.task_prefix_embeddings = self.torch.nn.Parameter(task)
        if condition in {CONDITION_F0, CONDITION_F1}:
            self.prefix_embeddings.requires_grad_(False)
        else:
            self.prefix_embeddings.requires_grad_(True)

        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("new-task transfer requires a completely frozen base model")
        if int(self.active_prefix_embeddings().shape[0]) != EFFECTIVE_PREFIX_LENGTH:
            raise RuntimeError("new-task transfer effective prefix length must be 32")

        self.initialization_audit = {
            "shared_behavior": shared_audit,
            "task_behavior": task_audit,
            "raw_shared_sha256": tensor_sha256(raw_shared),
            "selected_shared_sha256": tensor_sha256(shared),
            "task_sha256": tensor_sha256(task),
            "source_checkpoint": self.source_metadata or None,
        }

    def active_prefix_embeddings(self):
        if self.condition == CONDITION_I32:
            return _flatten_prefix_embeddings(self.prefix_embeddings)
        return self.torch.cat(
            [
                _flatten_prefix_embeddings(self.prefix_embeddings),
                self.task_prefix_embeddings,
            ],
            dim=0,
        )

    def shared_prefix_tensor(self):
        return self.prefix_embeddings[0]

    def task_prefix_tensor(self):
        if self.condition == CONDITION_I32:
            return self.prefix_embeddings[1]
        return self.task_prefix_embeddings

    def trainable_parameters(self) -> list[Any]:
        if self.condition == CONDITION_I32:
            return [self.prefix_embeddings]
        return [self.task_prefix_embeddings]

    def state_dict(self) -> dict[str, Any]:
        state = {
            "protocol_version": self.protocol_version,
            "condition": self.condition,
            "prefix_length": PREFIX_BLOCK_LENGTH,
            "effective_prefix_length": EFFECTIVE_PREFIX_LENGTH,
            "use_residual_reparameterization": False,
            "initialization_audit": self.initialization_audit,
        }
        if self.condition == CONDITION_I32:
            state["prefix_embeddings"] = self.prefix_embeddings.detach().cpu()
        else:
            state["shared_prefix_embeddings"] = self.prefix_embeddings.detach().cpu()
            state["task_prefix_embeddings"] = (
                self.task_prefix_embeddings.detach().cpu()
            )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("protocol_version")) != self.protocol_version:
            raise ValueError("checkpoint protocol does not match new-task transfer v1")
        if str(state.get("condition")) != self.condition:
            raise ValueError("checkpoint condition does not match configured condition")
        with self.torch.no_grad():
            if self.condition == CONDITION_I32:
                value = state["prefix_embeddings"].to(
                    device=self.device, dtype=self.prefix_embeddings.dtype
                )
                if tuple(value.shape) != tuple(self.prefix_embeddings.shape):
                    raise ValueError("I32 checkpoint prefix shape mismatch")
                self.prefix_embeddings.copy_(value)
            else:
                shared = state["shared_prefix_embeddings"].to(
                    device=self.device, dtype=self.prefix_embeddings.dtype
                )
                task = state["task_prefix_embeddings"].to(
                    device=self.device, dtype=self.task_prefix_embeddings.dtype
                )
                if tuple(shared.shape) != tuple(self.prefix_embeddings.shape):
                    raise ValueError("frozen Shared16 checkpoint shape mismatch")
                if tuple(task.shape) != tuple(self.task_prefix_embeddings.shape):
                    raise ValueError("Task16 checkpoint shape mismatch")
                self.prefix_embeddings.copy_(shared)
                self.task_prefix_embeddings.copy_(task)
