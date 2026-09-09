"""Functional prefix interventions for corrected v2 Shared+Task runs.

The helpers in this module construct inference-only prompt tensors.  They do
not mutate checkpoint parameters and never perform optimization.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

from skillopt.softprefix.model import ResidualPromptMLP
from skillopt.softprefix.multitask_v2 import (
    LAYOUT_SHARED_TASK,
    RESIDUAL_GLOBAL_MATCHED,
    JointV2SoftPrefixVisionLM,
)


DEFAULT_RESIDUAL_ALPHAS = (0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
RESTORE_FINAL = "final"
RESTORE_SHARED_INITIAL = "shared_initial"
RESTORE_TASK_INITIAL = "task_initial"
RESTORE_BOTH_INITIAL = "both_initial"
VALID_RESTORE_CONDITIONS = {
    RESTORE_FINAL,
    RESTORE_SHARED_INITIAL,
    RESTORE_TASK_INITIAL,
    RESTORE_BOTH_INITIAL,
}


def parse_residual_alphas(values: Iterable[float | str]) -> list[float]:
    """Return finite, unique residual scales in caller-provided order."""
    parsed: list[float] = []
    for value in values:
        alpha = float(value)
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError(f"residual alpha must be finite and non-negative, got {value!r}")
        if alpha not in parsed:
            parsed.append(alpha)
    if not parsed:
        raise ValueError("at least one residual alpha is required")
    return parsed


def scaled_residual_prompt(final_raw, residual_contribution, alpha: float):
    """Interpolate from the final raw prompt to its folded representation."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if tuple(final_raw.shape) != tuple(residual_contribution.shape):
        raise ValueError("raw and residual tensors must have identical shapes")
    return final_raw + alpha * residual_contribution


def split_shared_task(prompt, *, prefix_length: int):
    """Split a flat 32-token v2-D prompt into Shared and Task blocks."""
    prefix_length = int(prefix_length)
    if prompt.dim() != 2:
        raise ValueError("prompt must have shape [tokens, hidden_size]")
    if prefix_length < 1 or prompt.shape[0] != 2 * prefix_length:
        raise ValueError(
            f"expected exactly {2 * prefix_length} prompt tokens, got {prompt.shape[0]}"
        )
    return prompt[:prefix_length], prompt[prefix_length:]


def validate_shared_task_artifact(
    artifact: dict[str, Any],
    *,
    checkpoint_tasks: Iterable[str],
) -> list[str]:
    """Validate the four-state artifact needed for intervention construction."""
    if str(artifact.get("protocol_version")) != "v2":
        raise ValueError("prefix-state artifact is not protocol v2")
    if str(artifact.get("layout")) != LAYOUT_SHARED_TASK:
        raise ValueError("interventions require a shared16_task16 artifact")
    if str(artifact.get("residual_mode")) != RESIDUAL_GLOBAL_MATCHED:
        raise ValueError("interventions currently require global_matched residual mode")
    prefix_length = int(artifact.get("prefix_length", 0))
    if prefix_length < 1:
        raise ValueError("artifact prefix_length must be positive")
    task_order = [str(name) for name in artifact.get("task_order", [])]
    if not task_order or set(task_order) != set(str(name) for name in checkpoint_tasks):
        raise ValueError("artifact tasks do not match checkpoint tasks")
    states = artifact.get("states")
    required_states = {"initial", "final_raw", "residual_contribution", "folded"}
    if not isinstance(states, dict) or not required_states.issubset(states):
        raise ValueError("artifact is missing one or more required prefix states")
    for state_name in required_states:
        if set(states[state_name]) != set(task_order):
            raise ValueError(f"state {state_name!r} does not contain every task")
        for task_name in task_order:
            split_shared_task(states[state_name][task_name], prefix_length=prefix_length)

    reference = task_order[0]
    for state_name in required_states:
        reference_shared, _ = split_shared_task(
            states[state_name][reference], prefix_length=prefix_length
        )
        for task_name in task_order[1:]:
            shared, _ = split_shared_task(
                states[state_name][task_name], prefix_length=prefix_length
            )
            if not reference_shared.equal(shared):
                raise ValueError(
                    f"Shared block differs across tasks in state {state_name!r}"
                )
    return task_order


def construct_raw_intervention(
    artifact: dict[str, Any],
    *,
    target_task: str,
    condition: str = RESTORE_FINAL,
    donor_task: str | None = None,
):
    """Construct a raw Shared+Task prompt before the final residual MLP.

    ``donor_task`` requests an interchange intervention and is mutually
    exclusive with initialization restoration conditions.
    """
    states = artifact["states"]
    prefix_length = int(artifact["prefix_length"])
    target_task = str(target_task)
    if target_task not in states["final_raw"]:
        raise KeyError(f"unknown target task {target_task!r}")
    condition = str(condition)
    if condition not in VALID_RESTORE_CONDITIONS:
        raise ValueError(f"unknown restoration condition {condition!r}")
    if donor_task is not None and condition != RESTORE_FINAL:
        raise ValueError("Task interchange cannot be combined with initialization restoration")

    initial_shared, initial_task = split_shared_task(
        states["initial"][target_task], prefix_length=prefix_length
    )
    final_shared, final_task = split_shared_task(
        states["final_raw"][target_task], prefix_length=prefix_length
    )

    if donor_task is not None:
        donor_task = str(donor_task)
        if donor_task not in states["final_raw"]:
            raise KeyError(f"unknown donor task {donor_task!r}")
        _, final_task = split_shared_task(
            states["final_raw"][donor_task], prefix_length=prefix_length
        )

    shared = initial_shared if condition in {RESTORE_SHARED_INITIAL, RESTORE_BOTH_INITIAL} else final_shared
    task = initial_task if condition in {RESTORE_TASK_INITIAL, RESTORE_BOTH_INITIAL} else final_task
    import torch

    return torch.cat([shared, task], dim=0)


def apply_final_residual_with_alpha(model, raw_prompt, *, alpha: float):
    """Apply the checkpoint's final global residual map at a controlled scale."""
    if model.layout != LAYOUT_SHARED_TASK:
        raise ValueError("model must use shared16_task16 layout")
    if not model.use_residual_reparameterization:
        if float(alpha) != 0.0:
            raise ValueError("non-zero residual alpha requires a residual checkpoint")
        return raw_prompt
    if model.residual_mode != RESIDUAL_GLOBAL_MATCHED:
        raise ValueError("only global_matched residual checkpoints are supported")
    raw_prompt = raw_prompt.to(device=model.device, dtype=model.prefix_embeddings.dtype)
    folded = ResidualPromptMLP.apply(model.residual_mlp, raw_prompt)
    return scaled_residual_prompt(raw_prompt, folded - raw_prompt, alpha)


def prefix_diagnostics(torch, raw_prompt, deployed_prompt, *, eps: float = 1e-12) -> dict[str, float]:
    """Return norm and cosine diagnostics for one intervention tensor."""
    raw = raw_prompt.detach().float().reshape(-1)
    deployed = deployed_prompt.detach().float().reshape(-1)
    residual = deployed - raw
    raw_norm = float(raw.norm().item())
    residual_norm = float(residual.norm().item())
    deployed_norm = float(deployed.norm().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(raw.unsqueeze(0), deployed.unsqueeze(0)).item()
    )
    return {
        "raw_norm": raw_norm,
        "scaled_residual_norm": residual_norm,
        "deployed_norm": deployed_norm,
        "scaled_residual_over_raw": residual_norm / max(raw_norm, float(eps)),
        "raw_deployed_cosine": cosine,
    }


def select_global_alpha(rows: Iterable[dict[str, Any]], task_order: Iterable[str]) -> dict[str, Any]:
    """Select alpha by validation macro score, breaking exact ties downward."""
    tasks = [str(name) for name in task_order]
    by_alpha: dict[float, dict[str, float]] = {}
    for row in rows:
        if row.get("family") != "alpha":
            continue
        alpha = float(row["alpha"])
        by_alpha.setdefault(alpha, {})[str(row["target_task"])] = float(row["score"])
    candidates = []
    for alpha, scores in by_alpha.items():
        if set(scores) != set(tasks):
            continue
        candidates.append({
            "alpha": alpha,
            "macro_score": sum(scores[name] for name in tasks) / len(tasks),
            "task_scores": {name: scores[name] for name in tasks},
        })
    if not candidates:
        raise ValueError("no alpha has complete validation results for every task")
    candidates.sort(key=lambda item: (-item["macro_score"], item["alpha"]))
    return {"selected": candidates[0], "candidates": candidates}


class IntervenableJointV2SoftPrefixVisionLM(JointV2SoftPrefixVisionLM):
    """v2 model with an inference-only fixed-prefix override."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fixed_prefix_override = None

    def set_prefix_override(self, prefix) -> None:
        expected = self.active_raw_prefix_embeddings()
        if tuple(prefix.shape) != tuple(expected.shape):
            raise ValueError(
                f"override shape must be {tuple(expected.shape)}, got {tuple(prefix.shape)}"
            )
        self._fixed_prefix_override = prefix.detach().to(
            device=self.device,
            dtype=self.prefix_embeddings.dtype,
        )

    def clear_prefix_override(self) -> None:
        self._fixed_prefix_override = None

    def active_prefix_embeddings(self):
        if self._fixed_prefix_override is not None:
            return self._fixed_prefix_override
        return super().active_prefix_embeddings()
