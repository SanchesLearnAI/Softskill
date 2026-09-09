"""Corrected v2 protocol for joint multi-task SoftSkill experiments.

This module is intentionally separate from ``multitask_trainer.py``.  The
original implementation is kept unchanged so completed and in-flight v1
experiments remain reproducible.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
from typing import Any, Iterable

from tqdm import tqdm

from skillopt.config import load_config
from skillopt.softprefix.model import (
    ResidualPromptMLP,
    SoftPrefixVisionLM,
    _flatten_prefix_embeddings,
    _module_state_to_cpu,
)
from skillopt.softprefix.multitask_trainer import (
    JointTask,
    _evaluate_tasks,
    _load_task,
    _save_checkpoint,
    _save_folded_prefixes,
    progressive_learning_rates,
)
from skillopt.softprefix.trainer import (
    SoftPrefixSettings,
    _batch_to_tensors,
    _load_init_text,
    _set_seed,
)


PROTOCOL_VERSION = "v2"
PREFIX_STATE_SCHEMA_VERSION = 1
LAYOUT_INDEPENDENT = "independent_2x16"
LAYOUT_SHARED_TASK = "shared16_task16"
VALID_LAYOUTS = {LAYOUT_INDEPENDENT, LAYOUT_SHARED_TASK}
INIT_LEGACY_TASK_BLOCKS = "legacy_task_blocks"
INIT_BEHAVIOR_MARKDOWN = "behavior_markdown"
INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD = "hybrid_shared_behavior_task_hard"
SHARED_BEHAVIOR_INITIALIZATION_MODES = {
    INIT_BEHAVIOR_MARKDOWN,
    INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD,
}
VALID_INITIALIZATION_MODES = {
    INIT_LEGACY_TASK_BLOCKS,
    INIT_BEHAVIOR_MARKDOWN,
    INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD,
}
LR_UNIFORM = "uniform"
LR_PROGRESSIVE = "progressive"
LR_BEHAVIOR_TWO_STAGE = "behavior_two_stage"
VALID_LR_SCHEDULES = {LR_UNIFORM, LR_PROGRESSIVE, LR_BEHAVIOR_TWO_STAGE}
BEHAVIOR_STAGE_SHARED_WARMUP = "shared_warmup"
BEHAVIOR_STAGE_SHARED_TASK = "shared_task_joint"
RESIDUAL_GLOBAL_MATCHED = "global_matched"
RESIDUAL_BRANCH_DECOUPLED = "branch_decoupled"
VALID_RESIDUAL_MODES = {RESIDUAL_GLOBAL_MATCHED, RESIDUAL_BRANCH_DECOUPLED}


def split_markdown_token_embeddings(token_embeddings, *, prefix_length: int, num_blocks: int = 2):
    """Return consecutive, equally sized Markdown embedding blocks.

    Short Markdown skills are repeated deterministically, matching the legacy
    SoftSkill text-initialization behaviour.
    """
    if token_embeddings.dim() != 2:
        raise ValueError("token_embeddings must have shape [tokens, hidden_size]")
    if token_embeddings.shape[0] == 0:
        raise ValueError("cannot initialize a prefix from an empty Markdown skill")
    if int(prefix_length) < 1 or int(num_blocks) < 1:
        raise ValueError("prefix_length and num_blocks must be positive")
    total_length = int(prefix_length) * int(num_blocks)
    repeats = (total_length + token_embeddings.shape[0] - 1) // token_embeddings.shape[0]
    tiled = token_embeddings.repeat((repeats, 1))[:total_length]
    return tiled.reshape(int(num_blocks), int(prefix_length), token_embeddings.shape[-1])


def positionwise_shared_initialization(task_blocks, *, eps: float = 1e-8):
    """Build an order-invariant shared block and retain the typical token norm.

    A plain arithmetic mean can have a much smaller norm than the pretrained
    token embeddings.  We therefore preserve the mean direction but rescale
    every position to the mean norm of the contributing task embeddings.
    """
    if task_blocks.dim() != 3:
        raise ValueError("task_blocks must have shape [tasks, prefix_length, hidden_size]")
    if task_blocks.shape[0] < 2:
        raise ValueError("shared initialization requires at least two tasks")
    task_blocks_float = task_blocks.float()
    mean_block = task_blocks_float.mean(dim=0)
    target_norm = task_blocks_float.norm(dim=-1).mean(dim=0)
    mean_norm = mean_block.norm(dim=-1)
    scale = target_norm / mean_norm.clamp_min(float(eps))
    scaled = (mean_block * scale.unsqueeze(-1)).to(dtype=task_blocks.dtype)

    # Preserve task-order invariance even in the exact-cancellation edge case.
    fallback = task_blocks.new_zeros(mean_block.shape)
    valid = (mean_norm > float(eps)).unsqueeze(-1)
    return scaled.where(valid, fallback)


def compose_behavior_markdown_blocks(shared_block, task_blocks, *, layout: str):
    """Compose matched independent/shared layouts from explicit behavior blocks.

    In the independent layout every task stores its own copy of the same
    behavior-derived shared block.  In the shared layout that block is stored
    once.  This makes the two layouts use identical initialization information
    while differing only in whether the common behavior is actually shared.
    """
    layout = str(layout).strip().lower()
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"layout must be one of {sorted(VALID_LAYOUTS)}, got {layout!r}")
    if shared_block.dim() != 2:
        raise ValueError("shared behavior block must have shape [prefix_length, hidden_size]")
    if not task_blocks:
        raise ValueError("task behavior blocks must not be empty")
    for task_name, task_block in task_blocks.items():
        if tuple(task_block.shape) != tuple(shared_block.shape):
            raise ValueError(
                f"task behavior block {task_name!r} has shape {tuple(task_block.shape)}; "
                f"expected {tuple(shared_block.shape)}"
            )

    if layout == LAYOUT_INDEPENDENT:
        independent = {
            str(task_name): shared_block.new_empty((2, *shared_block.shape)).copy_(
                shared_block.unsqueeze(0).expand(2, -1, -1)
            )
            for task_name in task_blocks
        }
        for task_name, task_block in task_blocks.items():
            independent[str(task_name)][1].copy_(task_block)
        return None, independent

    shared = shared_block.detach().clone().unsqueeze(0)
    private = {
        str(task_name): task_block.detach().clone()
        for task_name, task_block in task_blocks.items()
    }
    return shared, private


def compose_hybrid_shared_behavior_task_hard_blocks(
    shared_behavior_block,
    task_hard_blocks,
    *,
    layout: str,
):
    """Pair explicit Shared behavior with each Hard Markdown's legacy block 2."""
    if str(layout).strip().lower() != LAYOUT_SHARED_TASK:
        raise ValueError(
            "hybrid_shared_behavior_task_hard requires shared16_task16"
        )
    private_blocks = {}
    for task_name, blocks in task_hard_blocks.items():
        if blocks.dim() != 3 or blocks.shape[0] != 2:
            raise ValueError(
                f"Hard Markdown blocks for {task_name!r} must have shape "
                "[2, prefix_length, hidden_size]"
            )
        private_blocks[str(task_name)] = blocks[1]
    return compose_behavior_markdown_blocks(
        shared_behavior_block,
        private_blocks,
        layout=layout,
    )


def effective_optimizer_steps(num_batches: int, accumulation: int) -> int:
    """Number of optimizer steps needed to consume one loader pass."""
    if int(num_batches) < 1:
        raise ValueError("num_batches must be positive")
    if int(accumulation) < 1:
        raise ValueError("accumulation must be positive")
    return math.ceil(int(num_batches) / int(accumulation))


def token_weighted_microbatch_scales(
    supervised_token_counts: Iterable[int],
    *,
    num_tasks: int,
) -> list[float]:
    """Return exact token-weighted scales with one macro share per task."""
    counts = [int(count) for count in supervised_token_counts]
    if int(num_tasks) < 1:
        raise ValueError("num_tasks must be positive")
    if not counts or any(count < 1 for count in counts):
        raise ValueError("every microbatch must contain at least one supervised token")
    total = sum(counts)
    return [count / (total * int(num_tasks)) for count in counts]


def supervised_token_count(torch, batch: dict[str, Any]) -> int:
    """Count labels that remain active after the causal-LM one-token shift."""
    labels = batch.get("labels")
    if labels is None:
        return 0
    labels = labels if hasattr(labels, "ne") else torch.as_tensor(labels)
    shifted_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    return int(shifted_labels.ne(-100).sum().item())


def batch_row_count(batch: dict[str, Any]) -> int:
    """Return the number of examples represented by one collated batch."""
    labels = batch.get("labels")
    if labels is None:
        return 0
    return int(labels.shape[0] if hasattr(labels, "shape") else len(labels))


def _tensor_sha256(tensor) -> str:
    payload = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _step1_gradient_audit(torch, gradient) -> dict[str, Any]:
    if gradient is None:
        return {"passed": False, "present": False, "finite": False, "norm": None}
    value = gradient.detach().float()
    finite = bool(torch.isfinite(value).all().item())
    norm = float(value.norm().cpu())
    return {
        "passed": bool(finite and norm > 0.0),
        "present": True,
        "finite": finite,
        "norm": norm,
    }


def _step1_change_audit(torch, before, after) -> dict[str, Any]:
    delta = after.detach().float().cpu() - before.detach().float().cpu()
    finite = bool(torch.isfinite(delta).all().item())
    max_abs = float(delta.abs().max())
    return {
        "passed": bool(finite and max_abs > 0.0),
        "finite": finite,
        "max_abs_change": max_abs,
    }


def _module_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_residual_mlp_deterministic(
    torch,
    *,
    embedding_dim: int,
    bottleneck_size: int,
    seed: int,
):
    """Build an MLP from an isolated RNG stream without perturbing global RNG."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        return ResidualPromptMLP.build(
            torch,
            int(embedding_dim),
            int(bottleneck_size),
        )


def _stable_task_seed(base_seed: int, task_name: str) -> int:
    task_digest = hashlib.sha256(str(task_name).encode("utf-8")).digest()
    offset = int.from_bytes(task_digest[:4], byteorder="big", signed=False)
    return (int(base_seed) + offset) % (2**31 - 1)


class JointV2SoftPrefixVisionLM(SoftPrefixVisionLM):
    """Matched 32-token models for the corrected v2 four-cell comparison.

    ``independent_2x16`` stores two Markdown blocks for each task.
    ``shared16_task16`` stores one symmetric shared block and the second
    Markdown block for each task.  Strict residual variants use one matched
    global reparameterizer; progressive Shared+Task runs use separate shared
    and task branches so their learning-rate schedules are fully decoupled.
    """

    def __init__(
        self,
        model_name: str,
        *,
        layout: str,
        prefix_length: int,
        task_init_texts: dict[str, str],
        initialization_mode: str = INIT_LEGACY_TASK_BLOCKS,
        shared_behavior_text: str = "",
        task_behavior_texts: dict[str, str] | None = None,
        residual_bottleneck_size: int = 400,
        use_residual_reparameterization: bool = False,
        residual_mode: str = RESIDUAL_GLOBAL_MATCHED,
        residual_init_seed: int = 1_000_003,
        torch_dtype: str = "auto",
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        layout = str(layout).strip().lower()
        if layout not in VALID_LAYOUTS:
            raise ValueError(f"layout must be one of {sorted(VALID_LAYOUTS)}, got {layout!r}")
        if len(task_init_texts) < 2:
            raise ValueError("v2 joint training requires at least two task Markdown skills")
        if any(not str(text).strip() for text in task_init_texts.values()):
            raise ValueError("every v2 task must have a non-empty Markdown initialization")
        initialization_mode = str(initialization_mode).strip().lower()
        if initialization_mode not in VALID_INITIALIZATION_MODES:
            raise ValueError(
                "initialization_mode must be one of "
                f"{sorted(VALID_INITIALIZATION_MODES)}, got {initialization_mode!r}"
            )
        task_behavior_texts = {
            str(name): str(text)
            for name, text in (task_behavior_texts or {}).items()
        }
        if initialization_mode in SHARED_BEHAVIOR_INITIALIZATION_MODES:
            if not str(shared_behavior_text).strip():
                raise ValueError(
                    f"{initialization_mode} initialization requires shared_behavior_text"
                )
        if initialization_mode == INIT_BEHAVIOR_MARKDOWN:
            if set(task_behavior_texts) != {str(name) for name in task_init_texts}:
                raise ValueError(
                    "behavior_markdown task texts must exactly match configured task names"
                )
            if any(not text.strip() for text in task_behavior_texts.values()):
                raise ValueError("every task behavior Markdown must be non-empty")
        if (
            initialization_mode == INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD
            and layout != LAYOUT_SHARED_TASK
        ):
            raise ValueError(
                "hybrid_shared_behavior_task_hard requires shared16_task16"
            )
        residual_mode = str(residual_mode).strip().lower()
        if residual_mode not in VALID_RESIDUAL_MODES:
            raise ValueError(
                f"residual_mode must be one of {sorted(VALID_RESIDUAL_MODES)}, "
                f"got {residual_mode!r}"
            )
        if residual_mode == RESIDUAL_BRANCH_DECOUPLED and layout != LAYOUT_SHARED_TASK:
            raise ValueError("branch_decoupled residuals require shared16_task16 layout")

        base_blocks = 2 if layout == LAYOUT_INDEPENDENT else 1
        super().__init__(
            model_name,
            prefix_length=prefix_length,
            num_soft_skills=base_blocks,
            init_text="",
            init_strategy="random",
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        self.protocol_version = PROTOCOL_VERSION
        self.layout = layout
        self.initialization_mode = initialization_mode
        self.initialization_spec = {
            "mode": initialization_mode,
            "task_init_texts": {
                str(name): str(text)
                for name, text in task_init_texts.items()
            },
            "shared_behavior_text": str(shared_behavior_text),
            "task_behavior_texts": dict(task_behavior_texts),
        }
        self.use_residual_reparameterization = bool(use_residual_reparameterization)
        self.residual_bottleneck_size = int(residual_bottleneck_size)
        self.residual_mode = residual_mode
        self.residual_init_seed = int(residual_init_seed)
        self.task_prefix_embeddings = self.torch.nn.ParameterDict()

        if initialization_mode == INIT_LEGACY_TASK_BLOCKS:
            blocks_by_task = {
                str(task_name): self._markdown_blocks(text)
                for task_name, text in task_init_texts.items()
            }
            if layout == LAYOUT_INDEPENDENT:
                task_initializations = blocks_by_task
                shared_initialization = None
            else:
                shared_candidates = self.torch.stack(
                    [blocks[0] for blocks in blocks_by_task.values()],
                    dim=0,
                )
                shared_initialization = positionwise_shared_initialization(shared_candidates).unsqueeze(0)
                task_initializations = {
                    task_name: blocks[1]
                    for task_name, blocks in blocks_by_task.items()
                }
        elif initialization_mode == INIT_BEHAVIOR_MARKDOWN:
            shared_behavior_block = self._markdown_blocks(
                shared_behavior_text,
                num_blocks=1,
            )[0]
            task_behavior_blocks = {
                task_name: self._markdown_blocks(text, num_blocks=1)[0]
                for task_name, text in task_behavior_texts.items()
            }
            shared_initialization, task_initializations = compose_behavior_markdown_blocks(
                shared_behavior_block,
                task_behavior_blocks,
                layout=layout,
            )
        else:
            shared_behavior_block = self._markdown_blocks(
                shared_behavior_text,
                num_blocks=1,
            )[0]
            task_hard_blocks = {
                str(task_name): self._markdown_blocks(text, num_blocks=2)
                for task_name, text in task_init_texts.items()
            }
            shared_initialization, task_initializations = (
                compose_hybrid_shared_behavior_task_hard_blocks(
                    shared_behavior_block,
                    task_hard_blocks,
                    layout=layout,
                )
            )

        if layout == LAYOUT_INDEPENDENT:
            initial_parameter = self.prefix_embeddings
            for index, (task_name, initialization) in enumerate(task_initializations.items()):
                if index == 0:
                    parameter = initial_parameter
                else:
                    parameter = self.torch.nn.Parameter(self.torch.empty_like(initial_parameter))
                with self.torch.no_grad():
                    parameter.copy_(initialization)
                self.task_prefix_embeddings[task_name] = parameter
        else:
            if shared_initialization is None:
                raise RuntimeError("shared layout initialization did not produce a shared block")
            with self.torch.no_grad():
                self.prefix_embeddings.copy_(shared_initialization)
            for task_name, initialization in task_initializations.items():
                parameter = self.torch.nn.Parameter(initialization.detach().clone())
                self.task_prefix_embeddings[task_name] = parameter

        self.active_task = next(iter(self.task_prefix_embeddings))
        if self.use_residual_reparameterization:
            embedding_dim = int(self.prefix_embeddings.shape[-1])
            if self.residual_mode == RESIDUAL_GLOBAL_MATCHED:
                self.residual_mlp = build_residual_mlp_deterministic(
                    self.torch,
                    embedding_dim=embedding_dim,
                    bottleneck_size=self.residual_bottleneck_size,
                    seed=self.residual_init_seed,
                ).to(device=self.device, dtype=self.prefix_embeddings.dtype)
            else:
                self.shared_residual_mlp = build_residual_mlp_deterministic(
                    self.torch,
                    embedding_dim=embedding_dim,
                    bottleneck_size=self.residual_bottleneck_size,
                    seed=self.residual_init_seed,
                ).to(device=self.device, dtype=self.prefix_embeddings.dtype)
                self.task_residual_mlps = self.torch.nn.ModuleDict()
                for task_name in self.task_prefix_embeddings:
                    self.task_residual_mlps[task_name] = build_residual_mlp_deterministic(
                        self.torch,
                        embedding_dim=embedding_dim,
                        bottleneck_size=self.residual_bottleneck_size,
                        seed=_stable_task_seed(self.residual_init_seed, task_name),
                    ).to(device=self.device, dtype=self.prefix_embeddings.dtype)

        self.initialization_audit = self._initialization_audit(
            task_init_texts,
            shared_behavior_text=shared_behavior_text,
            task_behavior_texts=task_behavior_texts,
        )

    def _initialization_audit(
        self,
        task_init_texts: dict[str, str],
        *,
        shared_behavior_text: str,
        task_behavior_texts: dict[str, str],
    ) -> dict[str, Any]:
        audit: dict[str, Any] = {
            "mode": self.initialization_mode,
            "markdown_sha256": {
                name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                for name, text in task_init_texts.items()
            },
            "task_prefix_sha256": {
                name: _tensor_sha256(parameter)
                for name, parameter in self.task_prefix_embeddings.items()
            },
            "residual_init_seed": self.residual_init_seed,
            "residual_mode": self.residual_mode,
        }
        if self.initialization_mode == INIT_LEGACY_TASK_BLOCKS:
            audit["tokenization"] = {
                "task_markdown": {
                    name: self._markdown_token_audit(
                        text,
                        selected_length=2 * self.prefix_length,
                    )
                    for name, text in task_init_texts.items()
                }
            }
        elif self.initialization_mode == INIT_BEHAVIOR_MARKDOWN:
            audit["shared_behavior_markdown_sha256"] = hashlib.sha256(
                shared_behavior_text.encode("utf-8")
            ).hexdigest()
            audit["task_behavior_markdown_sha256"] = {
                name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                for name, text in task_behavior_texts.items()
            }
            audit["tokenization"] = {
                "shared_behavior_markdown": self._markdown_token_audit(
                    shared_behavior_text,
                    selected_length=self.prefix_length,
                ),
                "task_behavior_markdown": {
                    name: self._markdown_token_audit(
                        text,
                        selected_length=self.prefix_length,
                    )
                    for name, text in task_behavior_texts.items()
                },
            }
        else:
            audit["shared_behavior_markdown_sha256"] = hashlib.sha256(
                shared_behavior_text.encode("utf-8")
            ).hexdigest()
            audit["hybrid_private_block"] = {
                "source": "task Hard Markdown from each task config",
                "token_offset": self.prefix_length,
                "token_length": self.prefix_length,
                "selection": "legacy Block2 (tokens 16-31 when prefix_length=16)",
            }
            audit["tokenization"] = {
                "shared_behavior_markdown": self._markdown_token_audit(
                    shared_behavior_text,
                    selected_length=self.prefix_length,
                ),
                "task_hard_markdown_block2": {
                    name: self._markdown_block_audit(text, block_index=1)
                    for name, text in task_init_texts.items()
                },
            }
        if self.layout == LAYOUT_SHARED_TASK:
            audit["shared_prefix_sha256"] = _tensor_sha256(self.prefix_embeddings)
        if self.use_residual_reparameterization:
            if self.residual_mode == RESIDUAL_GLOBAL_MATCHED:
                audit["residual_mlp_sha256"] = _module_sha256(self.residual_mlp)
            else:
                audit["shared_residual_mlp_sha256"] = _module_sha256(self.shared_residual_mlp)
                audit["task_residual_mlp_sha256"] = {
                    name: _module_sha256(module)
                    for name, module in self.task_residual_mlps.items()
                }
        return audit

    def _markdown_token_audit(self, text: str, *, selected_length: int) -> dict[str, Any]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        source_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
        if not source_ids:
            raise ValueError("tokenizer produced no tokens for a Markdown skill")
        repeats = (int(selected_length) + len(source_ids) - 1) // len(source_ids)
        selected_ids = (source_ids * repeats)[: int(selected_length)]
        return {
            "raw_text": text,
            "source_token_ids": source_ids,
            "source_token_count": len(source_ids),
            "selected_token_count": len(selected_ids),
            "repeated": len(source_ids) < int(selected_length),
            "truncated": len(source_ids) > int(selected_length),
            "selected_token_ids": selected_ids,
            "selected_text": self.tokenizer.decode(selected_ids),
        }

    def _markdown_block_audit(self, text: str, *, block_index: int) -> dict[str, Any]:
        block_index = int(block_index)
        if block_index < 0:
            raise ValueError("block_index must be non-negative")
        selected_offset = block_index * self.prefix_length
        full = self._markdown_token_audit(
            text,
            selected_length=selected_offset + self.prefix_length,
        )
        selected_ids = full["selected_token_ids"][selected_offset:]
        return {
            "raw_text": full["raw_text"],
            "source_token_ids": full["source_token_ids"],
            "source_token_count": full["source_token_count"],
            "selected_token_offset": selected_offset,
            "selected_token_count": len(selected_ids),
            "repeated": full["repeated"],
            "truncated": full["source_token_count"] > selected_offset + self.prefix_length,
            "selected_token_ids": selected_ids,
            "selected_text": self.tokenizer.decode(selected_ids),
        }

    def _markdown_blocks(self, text: str, *, num_blocks: int = 2):
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.numel() == 0:
            raise ValueError("tokenizer produced no tokens for a Markdown skill")
        with self.torch.no_grad():
            token_embeddings = self.model.get_input_embeddings()(input_ids)[0]
            token_embeddings = token_embeddings.to(dtype=self.prefix_embeddings.dtype)
            return split_markdown_token_embeddings(
                token_embeddings,
                prefix_length=self.prefix_length,
                num_blocks=num_blocks,
            )

    def set_active_task(self, task_name: str) -> None:
        if task_name not in self.task_prefix_embeddings:
            raise KeyError(f"unknown v2 task prefix: {task_name!r}")
        self.active_task = str(task_name)

    def active_prefix_embeddings(self):
        prompt = self.active_raw_prefix_embeddings()
        if not self.use_residual_reparameterization:
            return prompt
        if self.residual_mode == RESIDUAL_GLOBAL_MATCHED:
            return ResidualPromptMLP.apply(self.residual_mlp, prompt)
        shared = ResidualPromptMLP.apply(
            self.shared_residual_mlp,
            _flatten_prefix_embeddings(self.prefix_embeddings),
        )
        task = ResidualPromptMLP.apply(
            self.task_residual_mlps[self.active_task],
            self.task_prefix_embeddings[self.active_task],
        )
        return self.torch.cat([shared, task], dim=0)

    def active_raw_prefix_embeddings(self):
        """Return the trainable prompt before residual reparameterization."""
        if self.layout == LAYOUT_INDEPENDENT:
            return _flatten_prefix_embeddings(self.task_prefix_embeddings[self.active_task])
        shared = _flatten_prefix_embeddings(self.prefix_embeddings)
        task = self.task_prefix_embeddings[self.active_task]
        return self.torch.cat([shared, task], dim=0)

    def embedding_parameters(self) -> list[Any]:
        if self.layout == LAYOUT_INDEPENDENT:
            return list(self.task_prefix_embeddings.values())
        return [self.prefix_embeddings, *self.task_prefix_embeddings.values()]

    def shared_embedding_parameters(self) -> list[Any]:
        if self.layout != LAYOUT_SHARED_TASK:
            raise ValueError("independent layout has no shared prefix parameter")
        return [self.prefix_embeddings]

    def task_embedding_parameters(self) -> list[Any]:
        return list(self.task_prefix_embeddings.values())

    def reparameterization_parameters(self) -> list[Any]:
        if not self.use_residual_reparameterization:
            return []
        if self.residual_mode == RESIDUAL_GLOBAL_MATCHED:
            return list(self.residual_mlp.parameters())
        return (
            list(self.shared_residual_mlp.parameters())
            + list(self.task_residual_mlps.parameters())
        )

    def shared_reparameterization_parameters(self) -> list[Any]:
        if not self.use_residual_reparameterization:
            return []
        if self.residual_mode != RESIDUAL_BRANCH_DECOUPLED:
            raise ValueError("shared residual parameters require branch_decoupled mode")
        return list(self.shared_residual_mlp.parameters())

    def task_reparameterization_parameters(self) -> list[Any]:
        if not self.use_residual_reparameterization:
            return []
        if self.residual_mode != RESIDUAL_BRANCH_DECOUPLED:
            raise ValueError("task residual parameters require branch_decoupled mode")
        return list(self.task_residual_mlps.parameters())

    def trainable_parameters(self) -> list[Any]:
        return self.embedding_parameters() + self.reparameterization_parameters()

    def state_dict(self) -> dict[str, Any]:
        state = {
            "protocol_version": self.protocol_version,
            "layout": self.layout,
            "initialization_mode": getattr(
                self,
                "initialization_mode",
                INIT_LEGACY_TASK_BLOCKS,
            ),
            "prefix_length": self.prefix_length,
            "task_prefix_embeddings": {
                name: parameter.detach().cpu()
                for name, parameter in self.task_prefix_embeddings.items()
            },
            "residual_bottleneck_size": self.residual_bottleneck_size,
            "use_residual_reparameterization": self.use_residual_reparameterization,
            "residual_mode": self.residual_mode,
            "residual_init_seed": self.residual_init_seed,
            "active_task": self.active_task,
        }
        if hasattr(self, "initialization_spec"):
            state["initialization_spec"] = dict(self.initialization_spec)
        if self.layout == LAYOUT_SHARED_TASK:
            state["shared_prefix_embeddings"] = self.prefix_embeddings.detach().cpu()
        if self.use_residual_reparameterization:
            if self.residual_mode == RESIDUAL_GLOBAL_MATCHED:
                state["residual_mlp"] = _module_state_to_cpu(self.residual_mlp)
            else:
                state["shared_residual_mlp"] = _module_state_to_cpu(self.shared_residual_mlp)
                state["task_residual_mlps"] = {
                    name: _module_state_to_cpu(module)
                    for name, module in self.task_residual_mlps.items()
                }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("protocol_version")) != PROTOCOL_VERSION:
            raise ValueError("checkpoint is not a v2 joint SoftSkill checkpoint")
        if str(state.get("layout")) != self.layout:
            raise ValueError("checkpoint layout does not match the configured v2 layout")
        configured_initialization_mode = getattr(
            self,
            "initialization_mode",
            INIT_LEGACY_TASK_BLOCKS,
        )
        saved_initialization_mode = str(
            state.get("initialization_mode", INIT_LEGACY_TASK_BLOCKS)
        )
        if saved_initialization_mode != configured_initialization_mode:
            raise ValueError(
                "checkpoint initialization mode does not match the configured v2 model"
            )
        if int(state.get("prefix_length", -1)) != int(self.prefix_length):
            raise ValueError("checkpoint prefix length does not match the configured v2 model")
        saved_residual = bool(state.get("use_residual_reparameterization", False))
        if saved_residual != self.use_residual_reparameterization:
            raise ValueError("checkpoint residual flag does not match the configured v2 model")
        if str(state.get("residual_mode")) != self.residual_mode:
            raise ValueError("checkpoint residual mode does not match the configured v2 model")
        if int(state.get("residual_bottleneck_size", -1)) != self.residual_bottleneck_size:
            raise ValueError("checkpoint residual bottleneck does not match the configured v2 model")
        if int(state.get("residual_init_seed", -1)) != self.residual_init_seed:
            raise ValueError("checkpoint residual initialization seed does not match")
        task_state = state["task_prefix_embeddings"]
        if set(task_state) != set(self.task_prefix_embeddings):
            raise ValueError("checkpoint tasks do not match the configured v2 tasks")
        with self.torch.no_grad():
            for name, parameter in self.task_prefix_embeddings.items():
                value = task_state[name].to(device=self.device, dtype=parameter.dtype)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"task prefix shape mismatch for {name!r}")
                parameter.copy_(value)
            if self.layout == LAYOUT_SHARED_TASK:
                shared = state["shared_prefix_embeddings"].to(
                    device=self.device,
                    dtype=self.prefix_embeddings.dtype,
                )
                if tuple(shared.shape) != tuple(self.prefix_embeddings.shape):
                    raise ValueError("shared prefix shape mismatch")
                self.prefix_embeddings.copy_(shared)
        if self.use_residual_reparameterization:
            if self.residual_mode == RESIDUAL_GLOBAL_MATCHED:
                self.residual_mlp.load_state_dict(state["residual_mlp"])
            else:
                self.shared_residual_mlp.load_state_dict(state["shared_residual_mlp"])
                task_mlp_state = state["task_residual_mlps"]
                if set(task_mlp_state) != set(self.task_residual_mlps):
                    raise ValueError("checkpoint task residual MLPs do not match configured tasks")
                for name, module in self.task_residual_mlps.items():
                    module.load_state_dict(task_mlp_state[name])
        self.set_active_task(str(state.get("active_task", self.active_task)))


def capture_prefix_snapshot(
    model: JointV2SoftPrefixVisionLM,
    task_names: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Capture raw, residual-only, and folded prompts without changing model state."""
    original_task = model.active_task
    snapshot: dict[str, dict[str, Any]] = {
        "raw": {},
        "residual_contribution": {},
        "folded": {},
    }
    try:
        with model.torch.no_grad():
            for task_name in task_names:
                model.set_active_task(str(task_name))
                raw = model.active_raw_prefix_embeddings().detach().float().cpu().clone()
                folded = model.active_prefix_embeddings().detach().float().cpu().clone()
                if tuple(raw.shape) != tuple(folded.shape):
                    raise ValueError(
                        f"raw/folded shape mismatch for {task_name!r}: "
                        f"{tuple(raw.shape)} vs {tuple(folded.shape)}"
                    )
                snapshot["raw"][str(task_name)] = raw
                snapshot["residual_contribution"][str(task_name)] = folded - raw
                snapshot["folded"][str(task_name)] = folded
    finally:
        model.set_active_task(original_task)
    return snapshot


def _max_abs(values: Iterable[Any]) -> float:
    maxima = [float(value.detach().float().abs().max()) for value in values]
    return max(maxima, default=0.0)


def _shared_state_differences(
    states: dict[str, dict[str, Any]],
    task_names: list[str],
    prefix_length: int,
) -> dict[str, float]:
    if not task_names:
        return {}
    reference_task = task_names[0]
    differences: dict[str, float] = {}
    for state_name, task_values in states.items():
        reference = task_values[reference_task][:prefix_length]
        differences[state_name] = _max_abs(
            task_values[task_name][:prefix_length] - reference
            for task_name in task_names[1:]
        )
    return differences


def build_prefix_state_artifact(
    model: JointV2SoftPrefixVisionLM,
    task_names: Iterable[str],
    initial_snapshot: dict[str, dict[str, Any]],
    *,
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Build the auditable initial/raw/residual/folded four-state artifact."""
    ordered_tasks = [str(name) for name in task_names]
    final_snapshot = capture_prefix_snapshot(model, ordered_tasks)
    initial_raw = initial_snapshot["raw"]
    if set(initial_raw) != set(ordered_tasks):
        raise ValueError("initial prefix snapshot tasks do not match final tasks")
    states = {
        "initial": {name: initial_raw[name].clone() for name in ordered_tasks},
        "final_raw": {
            name: final_snapshot["raw"][name].clone() for name in ordered_tasks
        },
        "residual_contribution": {
            name: final_snapshot["residual_contribution"][name].clone()
            for name in ordered_tasks
        },
        "folded": {
            name: final_snapshot["folded"][name].clone() for name in ordered_tasks
        },
    }
    reconstruction_errors = {
        name: (
            states["folded"][name]
            - states["final_raw"][name]
            - states["residual_contribution"][name]
        ).abs().max()
        for name in ordered_tasks
    }
    initial_identity_errors = {
        name: (
            initial_snapshot["folded"][name] - initial_snapshot["raw"][name]
        ).abs().max()
        for name in ordered_tasks
    }
    first = states["initial"][ordered_tasks[0]]
    invariants: dict[str, Any] = {
        "folded_reconstruction_max_abs_error": _max_abs(reconstruction_errors.values()),
        "initial_residual_max_abs": _max_abs(initial_identity_errors.values()),
    }
    if model.layout == LAYOUT_SHARED_TASK:
        invariants["shared_max_abs_difference_across_tasks"] = _shared_state_differences(
            states,
            ordered_tasks,
            int(model.prefix_length),
        )
    return {
        "schema_version": PREFIX_STATE_SCHEMA_VERSION,
        "protocol_version": model.protocol_version,
        "layout": model.layout,
        "prefix_length": int(model.prefix_length),
        "effective_prefix_length": int(first.shape[0]),
        "hidden_size": int(first.shape[1]),
        "task_order": ordered_tasks,
        "use_residual_reparameterization": bool(model.use_residual_reparameterization),
        "residual_mode": model.residual_mode,
        "residual_bottleneck_size": int(model.residual_bottleneck_size),
        "residual_init_seed": int(model.residual_init_seed),
        "source_checkpoint": source_checkpoint,
        "state_definitions": {
            "initial": "raw Markdown-initialized prompt before the first optimizer step",
            "final_raw": "best-checkpoint trainable prompt before residual MLP",
            "residual_contribution": "Phi(final_raw) - final_raw",
            "folded": "final_raw + residual_contribution used for inference",
        },
        "states": states,
        "invariants": invariants,
    }


def save_initial_prefix_snapshot(
    torch,
    model: JointV2SoftPrefixVisionLM,
    task_names: Iterable[str],
    path: str,
) -> dict[str, dict[str, Any]]:
    """Persist initialization immediately so an interrupted run remains auditable."""
    ordered_tasks = [str(name) for name in task_names]
    snapshot = capture_prefix_snapshot(model, ordered_tasks)
    torch.save(
        {
            "schema_version": PREFIX_STATE_SCHEMA_VERSION,
            "phase": "before_training",
            "layout": model.layout,
            "prefix_length": int(model.prefix_length),
            "task_order": ordered_tasks,
            "snapshot": snapshot,
        },
        path,
    )
    return snapshot


def save_prefix_state_artifact(
    torch,
    model: JointV2SoftPrefixVisionLM,
    task_names: Iterable[str],
    initial_snapshot: dict[str, dict[str, Any]],
    path: str,
    *,
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    artifact = build_prefix_state_artifact(
        model,
        task_names,
        initial_snapshot,
        source_checkpoint=source_checkpoint,
    )
    torch.save(artifact, path)
    return artifact


def _task_accumulation(task: JointTask) -> int:
    accumulation = int(task.cfg.get("accumulation", 1) or 1)
    if accumulation < 1:
        raise ValueError(f"task {task.name!r} has invalid accumulation={accumulation}")
    return accumulation


def _next_batch(task: JointTask, iterators: dict[str, Any], wraps: dict[str, int]):
    try:
        return next(iterators[task.name])
    except StopIteration:
        wraps[task.name] += 1
        iterators[task.name] = iter(task.train_loader)
        return next(iterators[task.name])


def _optimizer_lrs(optimizer) -> dict[str, float]:
    return {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def behavior_warmup_steps(total_steps: int, warmup_fraction: float = 0.2) -> int:
    """Return the deterministic Shared-only step count for behavior v1."""
    total_steps = int(total_steps)
    warmup_fraction = float(warmup_fraction)
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0.0 < warmup_fraction < 1.0:
        raise ValueError("behavior warmup fraction must be between 0 and 1")
    return min(max(int(math.ceil(total_steps * warmup_fraction)), 1), total_steps)


def behavior_stage_for_step(
    global_step: int,
    *,
    total_steps: int,
    warmup_fraction: float = 0.2,
) -> str:
    """Map a zero-based optimizer-step index to the behavior training stage."""
    global_step = int(global_step)
    if global_step < 0 or global_step >= int(total_steps):
        raise ValueError("global_step must identify an optimizer step in this run")
    if global_step < behavior_warmup_steps(total_steps, warmup_fraction):
        return BEHAVIOR_STAGE_SHARED_WARMUP
    return BEHAVIOR_STAGE_SHARED_TASK


def set_behavior_training_stage(
    model: JointV2SoftPrefixVisionLM,
    optimizer,
    *,
    stage: str,
    shared_warmup_lr: float = 1e-3,
    shared_joint_lr: float = 1e-4,
    task_joint_lr: float = 1e-3,
    residual_joint_lr: float = 1e-3,
) -> dict[str, Any]:
    """Apply and describe one behavior-compression training stage.

    A global residual reparameterizer is kept frozen during the Shared-only
    warm-up.  This preserves the exact identity-start Markdown prefix and
    prevents the effective Task block from changing before stage two.
    """
    if model.layout != LAYOUT_SHARED_TASK:
        raise ValueError("behavior two-stage training requires shared16_task16")
    if model.initialization_mode not in SHARED_BEHAVIOR_INITIALIZATION_MODES:
        raise ValueError(
            "behavior two-stage training requires a Shared behavior initialization"
        )
    if (
        model.use_residual_reparameterization
        and model.residual_mode != RESIDUAL_GLOBAL_MATCHED
    ):
        raise ValueError(
            "behavior residual training requires one global_matched MLP shared "
            "across all virtual tokens and tasks"
        )
    if stage not in {BEHAVIOR_STAGE_SHARED_WARMUP, BEHAVIOR_STAGE_SHARED_TASK}:
        raise ValueError(f"unknown behavior training stage: {stage!r}")

    task_trainable = stage == BEHAVIOR_STAGE_SHARED_TASK
    residual_trainable = bool(
        model.use_residual_reparameterization
        and stage == BEHAVIOR_STAGE_SHARED_TASK
    )
    for parameter in model.shared_embedding_parameters():
        parameter.requires_grad_(True)
    for parameter in model.task_embedding_parameters():
        parameter.requires_grad_(task_trainable)
    for parameter in model.reparameterization_parameters():
        parameter.requires_grad_(residual_trainable)

    expected_groups = {"shared", "task"}
    if model.use_residual_reparameterization:
        expected_groups.add("residual")
    actual_groups = {str(group.get("name")) for group in optimizer.param_groups}
    if actual_groups != expected_groups:
        raise ValueError(
            "behavior two-stage optimizer groups do not match the configured paths; "
            f"got {sorted(actual_groups)}"
        )
    for group in optimizer.param_groups:
        if group.get("name") == "shared":
            group["lr"] = float(
                shared_warmup_lr
                if stage == BEHAVIOR_STAGE_SHARED_WARMUP
                else shared_joint_lr
            )
        elif group.get("name") == "task":
            group["lr"] = 0.0 if not task_trainable else float(task_joint_lr)
        elif group.get("name") == "residual":
            group["lr"] = (
                float(residual_joint_lr) if residual_trainable else 0.0
            )

    return {
        "stage": stage,
        "shared_trainable": all(
            parameter.requires_grad for parameter in model.shared_embedding_parameters()
        ),
        "task_trainable": {
            name: bool(parameter.requires_grad)
            for name, parameter in model.task_prefix_embeddings.items()
        },
        "residual_trainable": {
            "enabled": bool(model.use_residual_reparameterization),
            "all_parameters": (
                all(
                    parameter.requires_grad
                    for parameter in model.reparameterization_parameters()
                )
                if model.use_residual_reparameterization
                else False
            ),
        },
        "optimizer_lrs": _optimizer_lrs(optimizer),
        "use_residual_reparameterization": bool(model.use_residual_reparameterization),
    }


def _build_optimizer(
    torch,
    model: JointV2SoftPrefixVisionLM,
    *,
    learning_rate: float,
    lr_schedule: str,
    shared_lr_start: float,
):
    if lr_schedule == LR_UNIFORM:
        return torch.optim.AdamW(
            [{
                "params": model.trainable_parameters(),
                "lr": float(learning_rate),
                "weight_decay": 0.0,
                "name": "uniform",
            }]
        )
    if lr_schedule not in {LR_PROGRESSIVE, LR_BEHAVIOR_TWO_STAGE}:
        raise ValueError(f"lr_schedule must be one of {sorted(VALID_LR_SCHEDULES)}")
    if model.layout != LAYOUT_SHARED_TASK:
        raise ValueError(f"{lr_schedule} LR is only defined for shared16_task16")
    if lr_schedule == LR_BEHAVIOR_TWO_STAGE:
        if model.initialization_mode not in SHARED_BEHAVIOR_INITIALIZATION_MODES:
            raise ValueError(
                "behavior_two_stage requires a Shared behavior initialization"
            )
        if (
            model.use_residual_reparameterization
            and model.residual_mode != RESIDUAL_GLOBAL_MATCHED
        ):
            raise ValueError(
                "behavior_two_stage residual training requires global_matched mode"
            )
        groups = [
            {
                "params": model.shared_embedding_parameters(),
                "lr": float(shared_lr_start),
                "weight_decay": 0.0,
                "name": "shared",
            },
            {
                "params": model.task_embedding_parameters(),
                "lr": 0.0,
                "weight_decay": 0.0,
                "name": "task",
            },
        ]
        if model.use_residual_reparameterization:
            groups.append({
                "params": model.reparameterization_parameters(),
                "lr": 0.0,
                "weight_decay": 0.0,
                "name": "residual",
            })
        return torch.optim.AdamW(groups)
    if (
        model.use_residual_reparameterization
        and model.residual_mode != RESIDUAL_BRANCH_DECOUPLED
    ):
        raise ValueError(
            f"{lr_schedule} residual training requires branch_decoupled mode so "
            "the task path is fully frozen at task_lr=0"
        )
    groups = [
        {
            "params": (
                model.shared_embedding_parameters()
                + model.shared_reparameterization_parameters()
            ),
            "lr": float(shared_lr_start),
            "weight_decay": 0.0,
            "name": "shared",
        },
        {
            "params": (
                model.task_embedding_parameters()
                + model.task_reparameterization_parameters()
            ),
            "lr": 0.0,
            "weight_decay": 0.0,
            "name": "task",
        },
    ]
    return torch.optim.AdamW(groups)


def _set_progressive_lrs(
    optimizer,
    *,
    progress: float,
    shared_lr_start: float,
    shared_lr_min: float,
    shared_decay: float,
    task_lr_max: float,
    task_growth: float,
) -> None:
    shared_lr, task_lr = progressive_learning_rates(
        progress,
        shared_start=shared_lr_start,
        shared_min=shared_lr_min,
        shared_decay=shared_decay,
        task_max=task_lr_max,
        task_growth=task_growth,
    )
    for group in optimizer.param_groups:
        if group.get("name") == "shared":
            group["lr"] = shared_lr
        elif group.get("name") == "task":
            group["lr"] = task_lr


def _load_task_markdown(task_specs: Iterable[dict[str, str]]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for spec in task_specs:
        raw = load_config(spec["config"])
        settings = SoftPrefixSettings.from_dict(dict(raw.get("soft_prefix", {})))
        texts[str(spec["name"])] = _load_init_text(settings.init_text_path)
    return texts


def _task_markdown_source_paths(
    task_specs: Iterable[dict[str, str]],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for spec in task_specs:
        raw = load_config(spec["config"])
        settings = SoftPrefixSettings.from_dict(dict(raw.get("soft_prefix", {})))
        paths[str(spec["name"])] = settings.init_text_path
    return paths


def _load_behavior_markdown(
    task_specs: Iterable[dict[str, str]],
    *,
    shared_behavior_path: str,
) -> tuple[str, dict[str, str]]:
    shared_text = _load_init_text(shared_behavior_path).strip()
    if not shared_text.strip():
        raise FileNotFoundError(
            f"shared behavior Markdown is missing or empty: {shared_behavior_path!r}"
        )
    task_texts: dict[str, str] = {}
    for spec in task_specs:
        task_name = str(spec["name"])
        behavior_path = str(spec.get("behavior_init_path", ""))
        behavior_text = _load_init_text(behavior_path).strip()
        if not behavior_text.strip():
            raise FileNotFoundError(
                f"task behavior Markdown for {task_name!r} is missing or empty: "
                f"{behavior_path!r}"
            )
        task_texts[task_name] = behavior_text
    return shared_text, task_texts


def _load_json_audit(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"required audit file is missing: {path!r}")
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        content = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON audit file: {path!r}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"audit file must contain a JSON object: {path!r}")
    loaded = {
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content": content,
    }
    source_artifacts = content.get("source_artifacts", {})
    runtime_sources: dict[str, Any] = {}
    if isinstance(source_artifacts, dict):
        for name, source in source_artifacts.items():
            source_path = source.get("path") if isinstance(source, dict) else None
            if not source_path or not os.path.isfile(source_path):
                raise FileNotFoundError(
                    f"provenance source artifact is missing for {name!r}: {source_path!r}"
                )
            with open(source_path, "rb") as handle:
                source_raw = handle.read()
            runtime_sources[str(name)] = {
                "path": source_path,
                "sha256": hashlib.sha256(source_raw).hexdigest(),
                "byte_count": len(source_raw),
            }
    loaded["runtime_source_artifacts"] = runtime_sources
    return loaded


def _validate_behavior_tokenizer_audit(
    model: JointV2SoftPrefixVisionLM,
    static_audit: dict[str, Any],
    *,
    shared_behavior_path: str,
    task_specs: Iterable[dict[str, str]],
) -> None:
    files = static_audit.get("content", {}).get("files", {})
    runtime = model.initialization_audit["tokenization"]
    if model.initialization_mode == INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD:
        filename = os.path.basename(shared_behavior_path)
        expected = files.get(filename)
        if not isinstance(expected, dict):
            raise ValueError(
                "static behavior tokenizer audit is missing the Shared behavior file"
            )
        actual = runtime["shared_behavior_markdown"]
        checked_fields = (
            "raw_text",
            "source_token_ids",
            "source_token_count",
            "selected_token_ids",
            "selected_token_count",
            "selected_text",
            "repeated",
            "truncated",
        )
        mismatches = [
            field for field in checked_fields if expected.get(field) != actual.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"runtime tokenizer audit differs from static audit for {filename!r}: "
                f"{mismatches}"
            )
        return
    runtime_by_path = {
        os.path.basename(shared_behavior_path): runtime["shared_behavior_markdown"],
        **{
            os.path.basename(str(spec["behavior_init_path"])): runtime[
                "task_behavior_markdown"
            ][str(spec["name"])]
            for spec in task_specs
        },
    }
    if set(files) != set(runtime_by_path):
        raise ValueError(
            "static behavior tokenizer audit files do not match configured behavior files"
        )
    checked_fields = (
        "raw_text",
        "source_token_ids",
        "source_token_count",
        "selected_token_ids",
        "selected_token_count",
        "selected_text",
        "repeated",
        "truncated",
    )
    for filename, actual in runtime_by_path.items():
        expected = files[filename]
        mismatches = [
            field for field in checked_fields if expected.get(field) != actual.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"runtime tokenizer audit differs from static audit for {filename!r}: "
                f"{mismatches}"
            )


def initialization_kwargs_from_checkpoint_state(
    state: dict[str, Any],
    *,
    fallback_task_init_texts: dict[str, str],
) -> dict[str, Any]:
    """Recover constructor inputs while remaining compatible with old v2 states."""
    mode = str(state.get("initialization_mode", INIT_LEGACY_TASK_BLOCKS))
    spec = state.get("initialization_spec")
    if not isinstance(spec, dict):
        spec = {}
    task_init_texts = spec.get("task_init_texts")
    if not isinstance(task_init_texts, dict) or not task_init_texts:
        task_init_texts = fallback_task_init_texts
    kwargs: dict[str, Any] = {
        "task_init_texts": {
            str(name): str(text)
            for name, text in task_init_texts.items()
        },
        "initialization_mode": mode,
    }
    if mode in SHARED_BEHAVIOR_INITIALIZATION_MODES:
        shared_behavior_text = spec.get("shared_behavior_text")
        if not isinstance(shared_behavior_text, str) or not shared_behavior_text.strip():
            raise ValueError(
                "Shared behavior checkpoint does not embed shared behavior Markdown"
            )
        kwargs["shared_behavior_text"] = shared_behavior_text
    if mode == INIT_BEHAVIOR_MARKDOWN:
        task_behavior_texts = spec.get("task_behavior_texts")
        if not isinstance(task_behavior_texts, dict) or not task_behavior_texts:
            raise ValueError("behavior checkpoint does not embed task behavior Markdown")
        kwargs["task_behavior_texts"] = {
            str(name): str(text)
            for name, text in task_behavior_texts.items()
        }
    return kwargs


def _build_behavior_step1_preflight(
    model: JointV2SoftPrefixVisionLM,
    task_names: Iterable[str],
) -> dict[str, Any]:
    """Build fail-fast structural checks for the uniform behavior A/C pair."""
    torch = model.torch
    ordered_tasks = [str(name) for name in task_names]
    checks: dict[str, Any] = {}

    def record(name: str, passed: bool, **details: Any) -> None:
        checks[name] = {"passed": bool(passed), **details}

    initial_hashes: dict[str, str] = {}
    lengths: dict[str, int] = {}
    original_task = model.active_task
    try:
        for task_name in ordered_tasks:
            model.set_active_task(task_name)
            prompt = model.active_raw_prefix_embeddings().detach()
            lengths[task_name] = int(prompt.shape[0])
            initial_hashes[task_name] = _tensor_sha256(prompt)
    finally:
        model.set_active_task(original_task)

    record(
        "effective_prefix_length_is_32",
        all(value == 32 for value in lengths.values()),
        values=lengths,
    )
    record(
        "residual_disabled",
        not model.use_residual_reparameterization,
        value=bool(model.use_residual_reparameterization),
    )
    base_trainable = sum(
        parameter.numel()
        for parameter in model.model.parameters()
        if parameter.requires_grad
    )
    record("base_model_frozen", base_trainable == 0, trainable_parameters=base_trainable)

    if model.layout == LAYOUT_INDEPENDENT:
        parameters = [model.task_prefix_embeddings[name] for name in ordered_tasks]
        record(
            "a_shared_copies_equal_at_initialization",
            all(
                torch.equal(parameters[0][0].detach(), parameter[0].detach())
                for parameter in parameters[1:]
            ),
            shared_copy_sha256={
                name: _tensor_sha256(model.task_prefix_embeddings[name][0])
                for name in ordered_tasks
            },
        )
        data_ptrs = [int(parameter.data_ptr()) for parameter in parameters]
        record(
            "a_task_parameters_not_bound",
            len(set(data_ptrs)) == len(data_ptrs),
            parameter_data_ptrs=dict(zip(ordered_tasks, data_ptrs)),
        )
    elif model.layout == LAYOUT_SHARED_TASK:
        shared_parameters = model.shared_embedding_parameters()
        shared_ptr = int(model.prefix_embeddings.data_ptr())
        task_shared_ptrs = {name: shared_ptr for name in ordered_tasks}
        record(
            "c_tasks_reference_one_shared_parameter",
            len(shared_parameters) == 1
            and shared_parameters[0] is model.prefix_embeddings
            and len(set(task_shared_ptrs.values())) == 1,
            shared_parameter_count=len(shared_parameters),
            task_shared_data_ptrs=task_shared_ptrs,
        )
    else:
        record("known_behavior_pair_layout", False, layout=model.layout)

    return {
        "schema_version": 1,
        "phase": "preflight",
        "status": (
            "preflight_passed"
            if all(item["passed"] for item in checks.values())
            else "failed"
        ),
        "layout": model.layout,
        "task_order": ordered_tasks,
        "initial_active_prefix_sha256": initial_hashes,
        "checks": checks,
    }


def _write_step1_sanity(path: str, report: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def train_joint_soft_prefix_v2(
    *,
    task_specs: list[dict[str, str]],
    model_name: str,
    out_root: str,
    layout: str,
    initialization_mode: str = INIT_LEGACY_TASK_BLOCKS,
    shared_behavior_path: str = "",
    behavior_provenance_path: str = "skillopt/behavior_compression/v1/provenance.json",
    behavior_tokenizer_audit_path: str = (
        "skillopt/behavior_compression/v1/tokenizer_audit.json"
    ),
    seed: int = 1,
    prefix_length: int = 16,
    num_epochs: int = 3,
    expected_optimizer_steps: int | None = None,
    learning_rate: float = 1e-3,
    lr_schedule: str = "uniform",
    shared_lr_start: float = 1e-3,
    shared_lr_min: float = 5e-5,
    shared_decay: float = 3.0,
    task_lr_max: float = 1e-3,
    task_growth: float = 5.0,
    behavior_warmup_fraction: float = 0.2,
    behavior_shared_warmup_lr: float = 1e-3,
    behavior_shared_joint_lr: float = 1e-4,
    behavior_task_joint_lr: float = 1e-3,
    behavior_residual_joint_lr: float = 1e-3,
    residual_bottleneck_size: int = 400,
    use_residual_reparameterization: bool = False,
    residual_mode: str = RESIDUAL_GLOBAL_MATCHED,
    residual_init_seed: int | None = None,
) -> dict[str, Any]:
    """Train one corrected v2 cell with a fixed 32-token inference budget."""
    if len(task_specs) < 2:
        raise ValueError("v2 joint training requires at least two tasks")
    if int(num_epochs) < 1:
        raise ValueError("num_epochs must be at least 1")
    _set_seed(seed)
    os.makedirs(out_root, exist_ok=True)
    resolved_residual_seed = (
        int(residual_init_seed)
        if residual_init_seed is not None
        else int(seed) + 1_000_003
    )

    initialization_mode = str(initialization_mode).strip().lower()
    if initialization_mode not in VALID_INITIALIZATION_MODES:
        raise ValueError(
            "initialization_mode must be one of "
            f"{sorted(VALID_INITIALIZATION_MODES)}, got {initialization_mode!r}"
        )
    if lr_schedule not in VALID_LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {sorted(VALID_LR_SCHEDULES)}")
    behavior_provenance: dict[str, Any] | None = None
    behavior_tokenizer_audit: dict[str, Any] | None = None
    if lr_schedule == LR_BEHAVIOR_TWO_STAGE:
        if initialization_mode not in SHARED_BEHAVIOR_INITIALIZATION_MODES:
            raise ValueError(
                "behavior_two_stage requires a Shared behavior initialization"
            )
        if str(layout).strip().lower() != LAYOUT_SHARED_TASK:
            raise ValueError("behavior_two_stage requires shared16_task16")
        if (
            use_residual_reparameterization
            and str(residual_mode).strip().lower() != RESIDUAL_GLOBAL_MATCHED
        ):
            raise ValueError(
                "behavior_two_stage residual training requires global_matched mode"
            )
        behavior_warmup_steps(1, behavior_warmup_fraction)
        behavior_provenance = _load_json_audit(behavior_provenance_path)
        behavior_tokenizer_audit = _load_json_audit(behavior_tokenizer_audit_path)
    task_init_texts = _load_task_markdown(task_specs)
    task_init_paths = _task_markdown_source_paths(task_specs)
    shared_behavior_text = ""
    task_behavior_texts: dict[str, str] = {}
    if initialization_mode in SHARED_BEHAVIOR_INITIALIZATION_MODES:
        shared_behavior_text = _load_init_text(shared_behavior_path).strip()
        if not shared_behavior_text:
            raise FileNotFoundError(
                f"shared behavior Markdown is missing or empty: {shared_behavior_path!r}"
            )
    if initialization_mode == INIT_BEHAVIOR_MARKDOWN:
        shared_behavior_text, task_behavior_texts = _load_behavior_markdown(
            task_specs,
            shared_behavior_path=shared_behavior_path,
        )
    model = JointV2SoftPrefixVisionLM(
        model_name,
        layout=layout,
        prefix_length=prefix_length,
        task_init_texts=task_init_texts,
        initialization_mode=initialization_mode,
        shared_behavior_text=shared_behavior_text,
        task_behavior_texts=task_behavior_texts,
        residual_bottleneck_size=residual_bottleneck_size,
        use_residual_reparameterization=use_residual_reparameterization,
        residual_mode=residual_mode,
        residual_init_seed=resolved_residual_seed,
        torch_dtype="auto",
        device="auto",
        trust_remote_code=True,
    )
    torch = model.torch
    if behavior_tokenizer_audit is not None:
        _validate_behavior_tokenizer_audit(
            model,
            behavior_tokenizer_audit,
            shared_behavior_path=shared_behavior_path,
            task_specs=task_specs,
        )
    base_model_trainable_parameters = sum(
        parameter.numel()
        for parameter in model.model.parameters()
        if parameter.requires_grad
    )
    if base_model_trainable_parameters != 0:
        raise RuntimeError(
            "v2 protocol requires the complete base Qwen model to be frozen; "
            f"found {base_model_trainable_parameters} trainable base parameters"
        )
    tasks = [_load_task(spec, model=model, seed=seed) for spec in task_specs]
    task_names = [task.name for task in tasks]
    initial_prefix_states_path = os.path.join(out_root, "initial_prefix_states.pt")
    initial_prefix_snapshot = save_initial_prefix_snapshot(
        torch,
        model,
        task_names,
        initial_prefix_states_path,
    )
    accumulations = {task.name: _task_accumulation(task) for task in tasks}
    effective_steps = {
        task.name: effective_optimizer_steps(len(task.train_loader), accumulations[task.name])
        for task in tasks
    }
    cycles_per_epoch = max(effective_steps.values())
    total_steps = max(int(num_epochs) * cycles_per_epoch, 1)
    if (
        expected_optimizer_steps is not None
        and total_steps != int(expected_optimizer_steps)
    ):
        raise ValueError(
            f"expected {int(expected_optimizer_steps)} optimizer steps, "
            f"but resolved loaders/configuration produce {total_steps}"
        )
    optimizer = _build_optimizer(
        torch,
        model,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        shared_lr_start=(
            behavior_shared_warmup_lr
            if lr_schedule == LR_BEHAVIOR_TWO_STAGE
            else shared_lr_start
        ),
    )
    stage_events: list[dict[str, Any]] = []
    stage_step_counts = {
        BEHAVIOR_STAGE_SHARED_WARMUP: 0,
        BEHAVIOR_STAGE_SHARED_TASK: 0,
    }
    current_behavior_stage: str | None = None
    if lr_schedule == LR_BEHAVIOR_TWO_STAGE:
        current_behavior_stage = behavior_stage_for_step(
            0,
            total_steps=total_steps,
            warmup_fraction=behavior_warmup_fraction,
        )
        initial_stage_event = set_behavior_training_stage(
            model,
            optimizer,
            stage=current_behavior_stage,
            shared_warmup_lr=behavior_shared_warmup_lr,
            shared_joint_lr=behavior_shared_joint_lr,
            task_joint_lr=behavior_task_joint_lr,
            residual_joint_lr=behavior_residual_joint_lr,
        )
        initial_stage_event["global_step"] = 0
        initial_stage_event["reason"] = "training_start"
        stage_events.append(initial_stage_event)

    step1_sanity_enabled = bool(
        model.initialization_mode == INIT_BEHAVIOR_MARKDOWN
        and lr_schedule == LR_UNIFORM
        and not model.use_residual_reparameterization
        and model.layout in {LAYOUT_INDEPENDENT, LAYOUT_SHARED_TASK}
    )
    step1_sanity_path = os.path.join(out_root, "step1_sanity.json")
    step1_sanity: dict[str, Any] | None = None
    if step1_sanity_enabled:
        step1_sanity = _build_behavior_step1_preflight(model, task_names)
        _write_step1_sanity(step1_sanity_path, step1_sanity)
        if step1_sanity["status"] == "failed":
            raise AssertionError(f"behavior step-1 preflight failed; see {step1_sanity_path}")

    resolved_configs = {task.name: task.cfg for task in tasks}
    parameter_counts = {
        "embedding": sum(parameter.numel() for parameter in model.embedding_parameters()),
        "reparameterization": sum(
            parameter.numel() for parameter in model.reparameterization_parameters()
        ),
    }
    parameter_counts["total_trainable"] = sum(parameter_counts.values())
    protocol_audit = {
        "protocol_version": PROTOCOL_VERSION,
        "model_name": model_name,
        "model_revision": getattr(
            getattr(getattr(model, "model", None), "config", None),
            "_commit_hash",
            None,
        ),
        "layout": model.layout,
        "initialization_mode": model.initialization_mode,
        "initialization_source_paths": (
            {
                "shared_behavior": shared_behavior_path,
                "task_behavior": {
                    str(spec["name"]): str(spec.get("behavior_init_path", ""))
                    for spec in task_specs
                },
            }
            if model.initialization_mode == INIT_BEHAVIOR_MARKDOWN
            else (
                {
                    "shared_behavior": shared_behavior_path,
                    "task_hard_markdown": task_init_paths,
                }
                if model.initialization_mode
                == INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD
                else {
                    "task_markdown_from_config": {
                        str(spec["name"]): str(spec["config"])
                        for spec in task_specs
                    }
                }
            )
        ),
        "residual_mode": model.residual_mode,
        "use_residual_reparameterization": model.use_residual_reparameterization,
        "initialization": model.initialization_audit,
        "task_order": [task.name for task in tasks],
        "resolved_configs": resolved_configs,
        "accumulation": accumulations,
        "effective_loader_steps": effective_steps,
        "cycles_per_epoch": cycles_per_epoch,
        "parameter_counts": parameter_counts,
        "base_model_trainable_parameters": base_model_trainable_parameters,
        "training_schedule": (
            {
                "name": LR_BEHAVIOR_TWO_STAGE,
                "total_optimizer_steps": total_steps,
                "expected_optimizer_steps": expected_optimizer_steps,
                "warmup_fraction": float(behavior_warmup_fraction),
                "warmup_optimizer_steps": behavior_warmup_steps(
                    total_steps,
                    behavior_warmup_fraction,
                ),
                "joint_optimizer_steps": total_steps
                - behavior_warmup_steps(total_steps, behavior_warmup_fraction),
                "shared_warmup_lr": float(behavior_shared_warmup_lr),
                "shared_joint_lr": float(behavior_shared_joint_lr),
                "task_joint_lr": float(behavior_task_joint_lr),
                "residual_warmup_lr": 0.0,
                "residual_joint_lr": (
                    float(behavior_residual_joint_lr)
                    if model.use_residual_reparameterization
                    else None
                ),
                "residual_warmup_policy": (
                    "frozen_identity_start"
                    if model.use_residual_reparameterization
                    else None
                ),
                "optimizer_step_requires_all_tasks": True,
                "events": stage_events,
            }
            if lr_schedule == LR_BEHAVIOR_TWO_STAGE
            else {"name": lr_schedule}
        ),
        "behavior_source_provenance": behavior_provenance,
        "behavior_tokenizer_audit": behavior_tokenizer_audit,
        "step1_sanity": {
            "enabled": step1_sanity_enabled,
            "path": step1_sanity_path if step1_sanity_enabled else None,
        },
        "torch_version": str(torch.__version__),
        "transformers_version": importlib.metadata.version("transformers"),
    }
    with open(os.path.join(out_root, "protocol_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(protocol_audit, handle, ensure_ascii=False, indent=2)
    stage_events_path = os.path.join(out_root, "training_stage_events.json")
    if lr_schedule == LR_BEHAVIOR_TWO_STAGE:
        with open(stage_events_path, "w", encoding="utf-8") as handle:
            json.dump(stage_events, handle, ensure_ascii=False, indent=2)

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_path = os.path.join(out_root, "best_v2.pt")
    latest_path = os.path.join(out_root, "latest_v2.pt")
    global_step = 0

    for epoch in range(1, int(num_epochs) + 1):
        started = time.time()
        iterators = {task.name: iter(task.train_loader) for task in tasks}
        task_losses: dict[str, list[float]] = {task.name: [] for task in tasks}
        task_loss_token_sums = {task.name: 0.0 for task in tasks}
        microbatch_counts = {task.name: 0 for task in tasks}
        sample_counts = {task.name: 0 for task in tasks}
        supervised_token_counts = {task.name: 0 for task in tasks}
        wraps = {task.name: 0 for task in tasks}
        bar = tqdm(range(cycles_per_epoch), desc=f"Joint v2 {epoch}/{num_epochs}", unit="step")
        for _ in bar:
            progress = global_step / max(total_steps - 1, 1)
            if lr_schedule == LR_PROGRESSIVE:
                _set_progressive_lrs(
                    optimizer,
                    progress=progress,
                    shared_lr_start=shared_lr_start,
                    shared_lr_min=shared_lr_min,
                    shared_decay=shared_decay,
                    task_lr_max=task_lr_max,
                    task_growth=task_growth,
                )
            elif lr_schedule == LR_BEHAVIOR_TWO_STAGE:
                requested_stage = behavior_stage_for_step(
                    global_step,
                    total_steps=total_steps,
                    warmup_fraction=behavior_warmup_fraction,
                )
                if requested_stage != current_behavior_stage:
                    current_behavior_stage = requested_stage
                    stage_event = set_behavior_training_stage(
                        model,
                        optimizer,
                        stage=current_behavior_stage,
                        shared_warmup_lr=behavior_shared_warmup_lr,
                        shared_joint_lr=behavior_shared_joint_lr,
                        task_joint_lr=behavior_task_joint_lr,
                        residual_joint_lr=behavior_residual_joint_lr,
                    )
                    stage_event["global_step"] = global_step
                    stage_event["reason"] = "scheduled_transition"
                    stage_events.append(stage_event)
                    protocol_audit["training_schedule"]["events"] = stage_events
                    with open(
                        os.path.join(out_root, "protocol_audit.json"),
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(protocol_audit, handle, ensure_ascii=False, indent=2)
                    with open(stage_events_path, "w", encoding="utf-8") as handle:
                        json.dump(stage_events, handle, ensure_ascii=False, indent=2)
            step1_before: dict[str, Any] | None = None
            if step1_sanity_enabled and global_step == 0:
                if model.layout == LAYOUT_INDEPENDENT:
                    step1_before = {
                        f"shared/{name}": parameter[0].detach().clone()
                        for name, parameter in model.task_prefix_embeddings.items()
                    }
                    step1_before.update({
                        f"task/{name}": parameter[1].detach().clone()
                        for name, parameter in model.task_prefix_embeddings.items()
                    })
                else:
                    step1_before = {
                        "shared": model.prefix_embeddings[0].detach().clone(),
                        **{
                            f"task/{name}": parameter.detach().clone()
                            for name, parameter in model.task_prefix_embeddings.items()
                        },
                    }
            optimizer.zero_grad(set_to_none=True)

            # Fixed, auditable order.  One optimizer step is taken only after
            # every task has contributed its configured effective batch.
            for task in tasks:
                model.set_active_task(task.name)
                accumulation = accumulations[task.name]
                raw_batches = [
                    _next_batch(task, iterators, wraps)
                    for _microbatch in range(accumulation)
                ]
                active_counts = [
                    supervised_token_count(torch, batch)
                    for batch in raw_batches
                ]
                scales = token_weighted_microbatch_scales(
                    active_counts,
                    num_tasks=len(tasks),
                )
                for batch, active_count, scale in zip(raw_batches, active_counts, scales):
                    tensor_batch = _batch_to_tensors(torch, batch, model.device)
                    output = model.forward(tensor_batch)
                    if output.loss is None:
                        raise RuntimeError(f"task {task.name!r} returned no training loss")
                    (output.loss * scale).backward()
                    raw_loss = float(output.loss.detach().cpu())
                    task_losses[task.name].append(raw_loss)
                    task_loss_token_sums[task.name] += raw_loss * active_count
                    microbatch_counts[task.name] += 1
                    sample_counts[task.name] += batch_row_count(batch)
                    supervised_token_counts[task.name] += active_count
                    del tensor_batch, output

            if step1_sanity_enabled and global_step == 0:
                assert step1_sanity is not None
                gradient_checks: dict[str, Any] = {}
                base_gradients = sum(
                    1 for parameter in model.model.parameters() if parameter.grad is not None
                )
                gradient_checks["base_model_has_no_gradient"] = {
                    "passed": base_gradients == 0,
                    "parameters_with_gradient": base_gradients,
                }
                if model.layout == LAYOUT_INDEPENDENT:
                    for name, parameter in model.task_prefix_embeddings.items():
                        gradient = parameter.grad
                        gradient_checks[f"shared/{name}"] = _step1_gradient_audit(
                            torch,
                            None if gradient is None else gradient[0],
                        )
                        gradient_checks[f"task/{name}"] = _step1_gradient_audit(
                            torch,
                            None if gradient is None else gradient[1],
                        )
                else:
                    gradient_checks["shared"] = _step1_gradient_audit(
                        torch,
                        model.prefix_embeddings.grad,
                    )
                    for name, parameter in model.task_prefix_embeddings.items():
                        gradient_checks[f"task/{name}"] = _step1_gradient_audit(
                            torch,
                            parameter.grad,
                        )
                step1_sanity["gradient_checks_after_backward"] = gradient_checks
                gradients_passed = all(
                    item["passed"] for item in gradient_checks.values()
                )
                step1_sanity["phase"] = "backward"
                step1_sanity["status"] = (
                    "backward_passed" if gradients_passed else "failed"
                )
                _write_step1_sanity(step1_sanity_path, step1_sanity)
                if not gradients_passed:
                    raise AssertionError(
                        f"behavior step-1 gradient sanity failed; see {step1_sanity_path}"
                    )

            optimizer.step()
            if step1_sanity_enabled and global_step == 0:
                assert step1_sanity is not None and step1_before is not None
                change_checks: dict[str, Any] = {}
                if model.layout == LAYOUT_INDEPENDENT:
                    for name, parameter in model.task_prefix_embeddings.items():
                        change_checks[f"shared/{name}"] = _step1_change_audit(
                            torch,
                            step1_before[f"shared/{name}"],
                            parameter[0],
                        )
                        change_checks[f"task/{name}"] = _step1_change_audit(
                            torch,
                            step1_before[f"task/{name}"],
                            parameter[1],
                        )
                else:
                    change_checks["shared"] = _step1_change_audit(
                        torch,
                        step1_before["shared"],
                        model.prefix_embeddings[0],
                    )
                    for name, parameter in model.task_prefix_embeddings.items():
                        change_checks[f"task/{name}"] = _step1_change_audit(
                            torch,
                            step1_before[f"task/{name}"],
                            parameter,
                        )
                step1_sanity["parameter_change_checks_after_optimizer"] = change_checks
                changes_passed = all(item["passed"] for item in change_checks.values())
                step1_sanity["phase"] = "optimizer_step"
                step1_sanity["status"] = "passed" if changes_passed else "failed"
                step1_sanity["completed_global_step"] = 1
                _write_step1_sanity(step1_sanity_path, step1_sanity)
                if not changes_passed:
                    raise AssertionError(
                        f"behavior step-1 parameter-change sanity failed; see {step1_sanity_path}"
                    )
            if current_behavior_stage is not None:
                stage_step_counts[current_behavior_stage] += 1
            global_step += 1
            postfix = {name: f"{lr:.2e}" for name, lr in _optimizer_lrs(optimizer).items()}
            if current_behavior_stage is not None:
                postfix["stage"] = current_behavior_stage
            bar.set_postfix(postfix)

        joint_score, val_metrics = _evaluate_tasks(
            model,
            tasks,
            out_root=out_root,
            stage=f"epoch_{epoch:02d}_valid_seen",
            use_test=False,
        )
        record = {
            "protocol_version": PROTOCOL_VERSION,
            "epoch": epoch,
            "global_step": global_step,
            "joint_score": joint_score,
            "validation": val_metrics,
            "mean_train_loss": {
                name: task_loss_token_sums[name]
                / max(supervised_token_counts[name], 1)
                for name in task_losses
            },
            "microbatch_counts": microbatch_counts,
            "sample_counts": sample_counts,
            "supervised_token_counts": supervised_token_counts,
            "loader_wraps": wraps,
            "optimizer_lrs": _optimizer_lrs(optimizer),
            "training_stage": current_behavior_stage,
            "stage_step_counts": dict(stage_step_counts),
            "wall_time_s": round(time.time() - started, 1),
        }
        history.append(record)
        _save_checkpoint(
            torch,
            model,
            optimizer,
            latest_path,
            epoch=epoch,
            global_step=global_step,
            history=history,
        )
        if joint_score > best_score:
            best_score = joint_score
            _save_checkpoint(
                torch,
                model,
                optimizer,
                best_path,
                epoch=epoch,
                global_step=global_step,
                history=history,
            )
        with open(os.path.join(out_root, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)

    if lr_schedule == LR_BEHAVIOR_TWO_STAGE:
        protocol_audit["training_schedule"]["executed_stage_step_counts"] = dict(
            stage_step_counts
        )
        protocol_audit["training_schedule"]["executed_optimizer_steps"] = global_step
        with open(os.path.join(out_root, "protocol_audit.json"), "w", encoding="utf-8") as handle:
            json.dump(protocol_audit, handle, ensure_ascii=False, indent=2)

    best_state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_state["model"])
    folded_path = os.path.join(out_root, "folded_prefixes.pt")
    _save_folded_prefixes(torch, model, tasks, folded_path)
    prefix_states_path = os.path.join(out_root, "prefix_states.pt")
    prefix_state_artifact = save_prefix_state_artifact(
        torch,
        model,
        task_names,
        initial_prefix_snapshot,
        prefix_states_path,
        source_checkpoint=best_path,
    )
    _, test_metrics = _evaluate_tasks(
        model,
        tasks,
        out_root=out_root,
        stage="best_valid_unseen",
        use_test=True,
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "layout": model.layout,
        "initialization_mode": model.initialization_mode,
        "initialization": (
            {
                "independent": (
                    "one behavior-derived shared block copied per task plus one "
                    "behavior-derived private block"
                ),
                "shared_task": (
                    "one behavior-derived shared block stored once plus one "
                    "behavior-derived private block per task"
                ),
            }
            if model.initialization_mode == INIT_BEHAVIOR_MARKDOWN
            else (
                {
                    "shared_task": (
                        "explicit Shared behavior block plus each task Hard Markdown's "
                        "legacy Block2 (tokens 16-31)"
                    )
                }
                if model.initialization_mode
                == INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD
                else {
                    "independent": (
                        "two consecutive 16-token Markdown embedding blocks per task"
                    ),
                    "shared_task": (
                        "norm-calibrated positionwise mean of block 1; "
                        "task-specific block 2"
                    ),
                }
            )
        ),
        "task_order": [task.name for task in tasks],
        "accumulation": accumulations,
        "effective_loader_steps": effective_steps,
        "cycles_per_epoch": cycles_per_epoch,
        "loss_weighting": (
            "equal task macro weight; exact supervised-token weighting within each "
            "task's configured microbatches"
        ),
        "lr_schedule": lr_schedule,
        "training_schedule": protocol_audit["training_schedule"],
        "training_stage_events": (
            stage_events_path if lr_schedule == LR_BEHAVIOR_TWO_STAGE else None
        ),
        "step1_sanity": step1_sanity_path if step1_sanity_enabled else None,
        "residual_mode": model.residual_mode,
        "residual_init_seed": model.residual_init_seed,
        "best_joint_score": best_score,
        "best_checkpoint": best_path,
        "latest_checkpoint": latest_path,
        "folded_prefixes": folded_path,
        "initial_prefix_states": initial_prefix_states_path,
        "prefix_states": prefix_states_path,
        "prefix_state_invariants": prefix_state_artifact["invariants"],
        "test": test_metrics,
        "history": history,
        "prefix_length": prefix_length,
        "effective_prefix_length": prefix_length * 2,
        "storage": {
            "independent_unique_prefix_vectors": len(tasks) * prefix_length * 2,
            "shared_task_unique_prefix_vectors": prefix_length + len(tasks) * prefix_length,
            "current_layout_unique_prefix_vectors": (
                prefix_length + len(tasks) * prefix_length
                if model.layout == LAYOUT_SHARED_TASK
                else len(tasks) * prefix_length * 2
            ),
            "shared_task_reduction_fraction_vs_independent": (
                1.0
                - (prefix_length + len(tasks) * prefix_length)
                / (len(tasks) * prefix_length * 2)
            ),
        },
        "residual_bottleneck_size": residual_bottleneck_size,
        "use_residual_reparameterization": use_residual_reparameterization,
        "residual_mlp_sharing": (
            "none"
            if not model.use_residual_reparameterization
            else (
                "one matched MLP across all virtual tokens and tasks"
                if model.residual_mode == RESIDUAL_GLOBAL_MATCHED
                else "one shared-branch MLP plus one task-branch MLP per task"
            )
        ),
        "protocol_audit": protocol_audit,
        "seed": seed,
    }
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
