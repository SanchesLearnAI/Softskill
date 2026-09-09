"""Target-only F0/F1/I32 prefix structures built on the existing Qwen wrapper."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import (
    PREFIX_LENGTH,
    extract_shared_prefix,
    tensor_sha256,
)
from skillopt.softprefix.model import SoftPrefixVisionLM


PROTOCOL_VERSION = "new_task_transfer_v1"
CONDITION_F0 = "F0"
CONDITION_F1 = "F1"
CONDITION_I32 = "I32"
VALID_CONDITIONS = {CONDITION_F0, CONDITION_F1, CONDITION_I32}
VALID_TASKS = {"chartqa", "drop"}
EFFECTIVE_PREFIX_LENGTH = 32

ARTIFACT_ROOT = Path(__file__).resolve().parent
SHARED_BEHAVIOR_PATH = ARTIFACT_ROOT.parent.parent / "v1" / "shared_behavior.md"
SHARED_TOKEN_AUDIT_PATH = ARTIFACT_ROOT.parent.parent / "v1" / "tokenizer_audit.json"
TASK_TOKEN_AUDIT_PATH = ARTIFACT_ROOT / "tokenizer_audit.json"
TASK_BEHAVIOR_PATHS = {
    "chartqa": ARTIFACT_ROOT / "chartqa_behavior.md",
    "drop": ARTIFACT_ROOT / "drop_behavior.md",
}


def _normalize_condition(condition: str) -> str:
    normalized = str(condition).strip().upper()
    if normalized not in VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(VALID_CONDITIONS)}")
    return normalized


def _normalize_task(task_name: str) -> str:
    normalized = str(task_name).strip().lower()
    if normalized not in VALID_TASKS:
        raise ValueError(f"task_name must be one of {sorted(VALID_TASKS)}")
    return normalized


def _validate_block(name: str, tensor: Any) -> tuple[int, int]:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if tensor.dim() != 2 or int(tensor.shape[0]) != PREFIX_LENGTH:
        raise ValueError(
            f"{name} must have shape [16, hidden_size], got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    return int(tensor.shape[0]), int(tensor.shape[1])


class TransferPrefixParameters(torch.nn.Module):
    """Exactly one target task's 32-token parameters for F0, F1, or I32."""

    def __init__(
        self,
        raw_shared_block: Any,
        task_block: Any,
        *,
        condition: str,
        transferred_shared_block: Any | None = None,
    ) -> None:
        super().__init__()
        self.condition = _normalize_condition(condition)
        _, shared_hidden = _validate_block("raw_shared_block", raw_shared_block)
        _, task_hidden = _validate_block("task_block", task_block)
        if shared_hidden != task_hidden:
            raise ValueError("Shared16 and Task16 hidden sizes differ")
        if raw_shared_block.dtype != task_block.dtype:
            raise TypeError("Shared16 and Task16 dtypes differ")
        if raw_shared_block.device != task_block.device:
            raise ValueError("Shared16 and Task16 devices differ")

        if self.condition == CONDITION_F1:
            if transferred_shared_block is None:
                raise ValueError("F1 requires an explicitly extracted Shared16 tensor")
            _, transferred_hidden = _validate_block(
                "transferred_shared_block", transferred_shared_block
            )
            if transferred_hidden != shared_hidden:
                raise ValueError("transferred Shared16 hidden size differs")
            if transferred_shared_block.dtype != raw_shared_block.dtype:
                raise TypeError("transferred Shared16 dtype differs from model embeddings")
            if transferred_shared_block.device != raw_shared_block.device:
                raise ValueError("transferred Shared16 device differs from model embeddings")
            selected_shared = transferred_shared_block
        else:
            if transferred_shared_block is not None:
                raise ValueError("only F1 accepts a transferred Shared16 tensor")
            selected_shared = raw_shared_block

        self.hidden_size = shared_hidden
        if self.condition == CONDITION_I32:
            self.independent_prefix = torch.nn.Parameter(
                torch.cat([selected_shared, task_block], dim=0).detach().clone(),
                requires_grad=True,
            )
            self.register_parameter("shared_prefix", None)
            self.register_parameter("task_prefix", None)
        else:
            self.shared_prefix = torch.nn.Parameter(
                selected_shared.detach().clone(), requires_grad=False
            )
            self.task_prefix = torch.nn.Parameter(
                task_block.detach().clone(), requires_grad=True
            )
            self.register_parameter("independent_prefix", None)
        self.validate()

    def active_prefix_embeddings(self) -> Any:
        if self.condition == CONDITION_I32:
            return self.independent_prefix
        return torch.cat([self.shared_prefix, self.task_prefix], dim=0)

    def shared_prefix_tensor(self) -> Any:
        if self.condition == CONDITION_I32:
            return self.independent_prefix[:PREFIX_LENGTH]
        return self.shared_prefix

    def task_prefix_tensor(self) -> Any:
        if self.condition == CONDITION_I32:
            return self.independent_prefix[PREFIX_LENGTH:]
        return self.task_prefix

    def trainable_parameters(self) -> list[Any]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, value in self.named_parameters() if value.requires_grad]

    def validate(self) -> None:
        active = self.active_prefix_embeddings()
        if tuple(active.shape) != (EFFECTIVE_PREFIX_LENGTH, self.hidden_size):
            raise RuntimeError(
                "effective target prefix must be [32, hidden_size], got "
                f"{tuple(active.shape)}"
            )
        if self.condition == CONDITION_I32:
            if not self.independent_prefix.requires_grad:
                raise RuntimeError("I32 must train all 32 target-owned prefix vectors")
        else:
            if self.shared_prefix.requires_grad:
                raise RuntimeError("F0/F1 Shared16 must remain frozen")
            if not self.task_prefix.requires_grad:
                raise RuntimeError("F0/F1 Task16 must be trainable")

    def export_tensors(self) -> dict[str, Any]:
        if self.condition == CONDITION_I32:
            return {
                "independent_prefix_embeddings": self.independent_prefix.detach().cpu()
            }
        return {
            "shared_prefix_embeddings": self.shared_prefix.detach().cpu(),
            "task_prefix_embeddings": self.task_prefix.detach().cpu(),
        }

    def load_exported_tensors(self, state: dict[str, Any]) -> None:
        with torch.no_grad():
            if self.condition == CONDITION_I32:
                if set(state) != {"independent_prefix_embeddings"}:
                    raise ValueError("I32 state must contain only independent_prefix_embeddings")
                value = state["independent_prefix_embeddings"]
                self._copy_checked(self.independent_prefix, value, "I32 prefix")
            else:
                expected = {"shared_prefix_embeddings", "task_prefix_embeddings"}
                if set(state) != expected:
                    raise ValueError("F0/F1 state must contain exactly Shared16 and Task16")
                self._copy_checked(
                    self.shared_prefix,
                    state["shared_prefix_embeddings"],
                    "Shared16",
                )
                self._copy_checked(
                    self.task_prefix,
                    state["task_prefix_embeddings"],
                    "Task16",
                )
        self.validate()

    @staticmethod
    def _copy_checked(destination: Any, source: Any, name: str) -> None:
        if not torch.is_tensor(source):
            raise TypeError(f"saved {name} must be a tensor")
        if tuple(source.shape) != tuple(destination.shape):
            raise ValueError(
                f"saved {name} shape {tuple(source.shape)} does not match "
                f"{tuple(destination.shape)}"
            )
        if source.dtype != destination.dtype:
            raise TypeError(
                f"saved {name} dtype {source.dtype} does not match {destination.dtype}"
            )
        destination.copy_(source.to(device=destination.device))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_audit_record(audit_path: Path, filename: str) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if int(audit.get("selected_length", -1)) != PREFIX_LENGTH:
        raise ValueError(f"tokenizer audit at {audit_path} is not 16-token")
    record = audit.get("files", {}).get(filename)
    if not isinstance(record, dict):
        raise ValueError(f"tokenizer audit at {audit_path} has no {filename}")
    return record


def _embed_behavior_file(
    *,
    tokenizer: Any,
    embedding_layer: Any,
    behavior_path: Path,
    audit_path: Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    text = behavior_path.read_text(encoding="utf-8").strip()
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"]
    token_ids = [int(value) for value in input_ids[0].tolist()]
    if len(token_ids) != PREFIX_LENGTH:
        raise ValueError(
            f"{behavior_path.name} must tokenize to exactly 16 tokens, got {len(token_ids)}"
        )
    static_record = _read_audit_record(audit_path, behavior_path.name)
    expected_ids = [int(value) for value in static_record["selected_token_ids"]]
    if token_ids != expected_ids:
        raise ValueError(
            f"runtime tokenizer IDs for {behavior_path.name} differ from tokenizer_audit.json"
        )
    with torch.no_grad():
        block = embedding_layer(input_ids.to(device))[0].detach().clone()
    _validate_block(behavior_path.name, block)
    return block, {
        "path": str(behavior_path),
        "sha256": _file_sha256(behavior_path),
        "raw_text": text,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "selected_text": tokenizer.decode(token_ids),
        "embedding_sha256": tensor_sha256(block),
    }


def embed_task_behavior_blocks(
    *,
    tokenizer: Any,
    embedding_layer: Any,
    device: Any,
    task_name: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Embed the fixed Shared text and the selected target task's own text."""
    task_name = _normalize_task(task_name)
    shared, shared_audit = _embed_behavior_file(
        tokenizer=tokenizer,
        embedding_layer=embedding_layer,
        behavior_path=SHARED_BEHAVIOR_PATH,
        audit_path=SHARED_TOKEN_AUDIT_PATH,
        device=device,
    )
    task_path = TASK_BEHAVIOR_PATHS[task_name]
    task, task_audit = _embed_behavior_file(
        tokenizer=tokenizer,
        embedding_layer=embedding_layer,
        behavior_path=task_path,
        audit_path=TASK_TOKEN_AUDIT_PATH,
        device=device,
    )
    if shared.dtype != task.dtype or shared.shape != task.shape:
        raise ValueError("Shared and target behavior embeddings are incompatible")
    return shared, task, {
        "task_name": task_name,
        "shared_behavior": shared_audit,
        "task_behavior": task_audit,
    }


class TargetOnlyTransferSoftPrefixVisionLM(SoftPrefixVisionLM):
    """Frozen Qwen plus one target task's F0, F1, or I32 prefix."""

    def __init__(
        self,
        model_name: str,
        *,
        task_name: str,
        condition: str,
        source_checkpoint_path: str | os.PathLike[str] | None = None,
        torch_dtype: str = "auto",
        device: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        condition = _normalize_condition(condition)
        task_name = _normalize_task(task_name)
        explicit_source = str(source_checkpoint_path or "").strip()
        if condition == CONDITION_F1 and not explicit_source:
            raise ValueError("F1 requires an explicit source_checkpoint_path")
        if condition != CONDITION_F1 and explicit_source:
            raise ValueError("only F1 accepts source_checkpoint_path")

        super().__init__(
            model_name,
            prefix_length=PREFIX_LENGTH,
            num_soft_skills=1,
            init_text="",
            init_strategy="random",
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        embedding_layer = self.model.get_input_embeddings()
        raw_shared, task, initialization_audit = embed_task_behavior_blocks(
            tokenizer=self.tokenizer,
            embedding_layer=embedding_layer,
            device=self.device,
            task_name=task_name,
        )
        transferred = None
        source_metadata: dict[str, Any] = {}
        if condition == CONDITION_F1:
            transferred_cpu, source_metadata = extract_shared_prefix(
                explicit_source,
                expected_hidden_size=int(embedding_layer.embedding_dim),
                expected_dtype=embedding_layer.weight.dtype,
                torch_module=self.torch,
            )
            transferred = transferred_cpu.to(device=self.device)

        # The parent creates one generic random prefix.  It is replaced, not
        # trained alongside the condition-specific parameterization.
        del self.prefix_embeddings
        self.prefix_parameters = TransferPrefixParameters(
            raw_shared,
            task,
            condition=condition,
            transferred_shared_block=transferred,
        )
        self.model_name = str(model_name)
        self.task_name = task_name
        self.condition = condition
        self.use_residual_reparameterization = False
        self.initialization_audit = initialization_audit
        self.source_metadata = source_metadata
        self.validate_invariants()

    def active_prefix_embeddings(self) -> Any:
        return self.prefix_parameters.active_prefix_embeddings()

    def shared_prefix_tensor(self) -> Any:
        return self.prefix_parameters.shared_prefix_tensor()

    def task_prefix_tensor(self) -> Any:
        return self.prefix_parameters.task_prefix_tensor()

    def trainable_parameters(self) -> list[Any]:
        return self.prefix_parameters.trainable_parameters()

    def trainable_parameter_names(self) -> list[str]:
        return self.prefix_parameters.trainable_parameter_names()

    def validate_invariants(self) -> None:
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Qwen base model must be completely frozen")
        if self.use_residual_reparameterization:
            raise RuntimeError("new-task transfer v1 forbids residual reparameterization")
        self.prefix_parameters.validate()
        if int(self.active_prefix_embeddings().shape[0]) != EFFECTIVE_PREFIX_LENGTH:
            raise RuntimeError("effective inference prefix must contain exactly 32 tokens")

    def state_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "model_name": self.model_name,
            "task_name": self.task_name,
            "condition": self.condition,
            "prefix_length": PREFIX_LENGTH,
            "effective_prefix_length": EFFECTIVE_PREFIX_LENGTH,
            "use_residual_reparameterization": False,
            "prefix_tensors": self.prefix_parameters.export_tensors(),
            "initialization_audit": self.initialization_audit,
            "source_checkpoint": self.source_metadata or None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("protocol_version")) != PROTOCOL_VERSION:
            raise ValueError("checkpoint protocol does not match new-task transfer v1")
        if str(state.get("model_name")) != self.model_name:
            raise ValueError("checkpoint base model does not match configured model")
        if str(state.get("task_name")) != self.task_name:
            raise ValueError("checkpoint target task does not match configured task")
        if str(state.get("condition")) != self.condition:
            raise ValueError("checkpoint condition does not match configured condition")
        if int(state.get("prefix_length", -1)) != PREFIX_LENGTH:
            raise ValueError("checkpoint prefix_length must be 16")
        if int(state.get("effective_prefix_length", -1)) != EFFECTIVE_PREFIX_LENGTH:
            raise ValueError("checkpoint effective prefix length must be 32")
        if bool(state.get("use_residual_reparameterization", False)):
            raise ValueError("checkpoint unexpectedly enables residual")
        tensors = state.get("prefix_tensors")
        if not isinstance(tensors, dict):
            raise ValueError("checkpoint has no prefix_tensors state")
        self.prefix_parameters.load_exported_tensors(tensors)
        self.initialization_audit = dict(state.get("initialization_audit") or {})
        self.source_metadata = dict(state.get("source_checkpoint") or {})
        self.validate_invariants()
