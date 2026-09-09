#!/usr/bin/env python3
"""Attribute prefix similarity to initialization, training, and residual MLP.

The input files are small ``prefix_states.pt`` artifacts.  This analysis is
CPU-only and never loads the frozen language model or any dataset.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_prefix_similarity import (
    _comparison_row,
    _matrix,
    _safe_torch_load,
    _write_csv,
    aggregate_rows,
    similarity_metrics,
)


REQUIRED_STATES = (
    "initial",
    "final_raw",
    "residual_contribution",
    "folded",
)

DECOMPOSITION_METRICS = (
    "initial_norm",
    "final_raw_norm",
    "training_delta_norm",
    "residual_contribution_norm",
    "folded_norm",
    "total_delta_norm",
    "training_delta_over_initial",
    "residual_over_final_raw",
    "total_delta_over_initial",
    "training_magnitude_share",
    "residual_magnitude_share",
    "initial_final_raw_cosine",
    "final_raw_folded_cosine",
    "initial_folded_cosine",
    "training_residual_cosine",
    "folded_reconstruction_max_abs_error",
)


def _load_artifact(path: Path, expected_layout: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = _safe_torch_load(path)
    if int(artifact.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported prefix-state schema in {path}")
    if artifact.get("layout") != expected_layout:
        raise ValueError(
            f"expected layout {expected_layout!r}, got {artifact.get('layout')!r}"
        )
    states = artifact.get("states", {})
    missing = [name for name in REQUIRED_STATES if name not in states]
    if missing:
        raise ValueError(f"{path} is missing states: {missing}")
    tasks = [str(name) for name in artifact.get("task_order", [])]
    if len(tasks) < 2 or len(tasks) != len(set(tasks)):
        raise ValueError(f"invalid task order in {path}: {tasks}")
    expected_shape = (
        int(artifact["effective_prefix_length"]),
        int(artifact["hidden_size"]),
    )
    for state_name in REQUIRED_STATES:
        if set(states[state_name]) != set(tasks):
            raise ValueError(f"{state_name} task set does not match task_order")
        for task in tasks:
            value = _matrix(states[state_name][task])
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{state_name}/{task}: expected {expected_shape}, got {tuple(value.shape)}"
                )
    for task in tasks:
        error = (
            _matrix(states["folded"][task])
            - _matrix(states["final_raw"][task])
            - _matrix(states["residual_contribution"][task])
        ).abs().max()
        if float(error) > 1e-5:
            raise ValueError(f"folded reconstruction failed for {path}:{task}: {float(error)}")
    if expected_layout == "shared16_task16":
        prefix_length = int(artifact["prefix_length"])
        reference_task = tasks[0]
        for state_name in REQUIRED_STATES:
            reference = _matrix(states[state_name][reference_task])[:prefix_length]
            for task in tasks[1:]:
                difference = float(
                    (_matrix(states[state_name][task])[:prefix_length] - reference)
                    .abs()
                    .max()
                )
                if difference > 1e-5:
                    raise ValueError(
                        f"shared block differs across tasks in {state_name}: {difference}"
                    )
    artifact["_path"] = str(path.resolve())
    return artifact


def _common_tasks(
    independent: dict[str, Any],
    shared: dict[str, Any],
    requested: list[str],
) -> list[str]:
    available = set(independent["task_order"]) & set(shared["task_order"])
    ordered = requested or list(independent["task_order"])
    tasks = [task for task in ordered if task in available]
    if len(tasks) < 2:
        raise ValueError(f"need at least two common tasks, found {sorted(available)}")
    return tasks


def _state_components(
    artifact: dict[str, Any],
    state_name: str,
    task: str,
) -> dict[str, torch.Tensor]:
    full = _matrix(artifact["states"][state_name][task])
    prefix_length = int(artifact["prefix_length"])
    if full.shape[0] != 2 * prefix_length:
        raise ValueError(f"expected a 2x{prefix_length} prompt, got {tuple(full.shape)}")
    if artifact["layout"] == "independent_2x16":
        return {
            "full": full,
            "block1": full[:prefix_length],
            "block2": full[prefix_length:],
        }
    return {
        "full": full,
        "shared": full[:prefix_length],
        "private": full[prefix_length:],
    }


def _norm(value: torch.Tensor) -> float:
    return float(_matrix(value).norm())


def _ratio(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    return float(numerator) / max(float(denominator), eps)


def build_decomposition_rows(
    experiment: str,
    artifact: dict[str, Any],
    tasks: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        initial_parts = _state_components(artifact, "initial", task)
        final_parts = _state_components(artifact, "final_raw", task)
        residual_parts = _state_components(artifact, "residual_contribution", task)
        folded_parts = _state_components(artifact, "folded", task)
        for component in initial_parts:
            initial = initial_parts[component]
            final_raw = final_parts[component]
            residual = residual_parts[component]
            folded = folded_parts[component]
            training_delta = final_raw - initial
            total_delta = folded - initial
            initial_norm = _norm(initial)
            final_norm = _norm(final_raw)
            training_norm = _norm(training_delta)
            residual_norm = _norm(residual)
            folded_norm = _norm(folded)
            total_norm = _norm(total_delta)
            update_norm_sum = training_norm + residual_norm
            rows.append({
                "experiment": experiment,
                "layout": artifact["layout"],
                "task": task,
                "component": component,
                "tokens": int(_matrix(initial).shape[0]),
                "hidden_size": int(_matrix(initial).shape[1]),
                "initial_norm": initial_norm,
                "final_raw_norm": final_norm,
                "training_delta_norm": training_norm,
                "residual_contribution_norm": residual_norm,
                "folded_norm": folded_norm,
                "total_delta_norm": total_norm,
                "training_delta_over_initial": _ratio(training_norm, initial_norm),
                "residual_over_final_raw": _ratio(residual_norm, final_norm),
                "total_delta_over_initial": _ratio(total_norm, initial_norm),
                "training_magnitude_share": _ratio(training_norm, update_norm_sum),
                "residual_magnitude_share": _ratio(residual_norm, update_norm_sum),
                "initial_final_raw_cosine": similarity_metrics(initial, final_raw)["cosine_flat"],
                "final_raw_folded_cosine": similarity_metrics(final_raw, folded)["cosine_flat"],
                "initial_folded_cosine": similarity_metrics(initial, folded)["cosine_flat"],
                "training_residual_cosine": similarity_metrics(training_delta, residual)["cosine_flat"],
                "folded_reconstruction_max_abs_error": float(
                    (folded - final_raw - residual).abs().max()
                ),
            })
    return rows


def build_transition_rows(
    experiment: str,
    artifact: dict[str, Any],
    tasks: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transition_specs = (
        ("initialization_retention", "initial", "final_raw"),
        ("residual_effect", "final_raw", "folded"),
        ("end_to_end", "initial", "folded"),
        ("raw_vs_residual", "final_raw", "residual_contribution"),
        ("training_delta_vs_residual", "training_delta", "residual_contribution"),
    )
    for task in tasks:
        values = {
            state: _state_components(artifact, state, task)
            for state in REQUIRED_STATES
        }
        values["training_delta"] = {
            component: values["final_raw"][component] - values["initial"][component]
            for component in values["initial"]
        }
        for component in values["initial"]:
            for transition, left_state, right_state in transition_specs:
                row = _comparison_row(
                    transition,
                    experiment,
                    component,
                    f"{task}:{left_state}",
                    f"{task}:{right_state}",
                    values[left_state][component],
                    values[right_state][component],
                )
                row["task"] = task
                row["left_state"] = left_state
                row["right_state"] = right_state
                rows.append(row)
    return rows


def aggregate_decomposition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["experiment"], row["layout"], row["component"])
        groups.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (experiment, layout, component), values in sorted(groups.items()):
        item: dict[str, Any] = {
            "experiment": experiment,
            "layout": layout,
            "component": component,
            "task_count": len(values),
        }
        for metric in DECOMPOSITION_METRICS:
            metric_values = [float(value[metric]) for value in values]
            item[f"{metric}_mean"] = sum(metric_values) / len(metric_values)
            item[f"{metric}_min"] = min(metric_values)
            item[f"{metric}_max"] = max(metric_values)
        aggregates.append(item)
    return aggregates


def build_structure_rows(
    independent: dict[str, Any],
    shared: dict[str, Any],
    tasks: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_task = tasks[0]
    for state_name in REQUIRED_STATES:
        independent_parts = {
            task: _state_components(independent, state_name, task) for task in tasks
        }
        shared_parts = {
            task: _state_components(shared, state_name, task) for task in tasks
        }

        for component in ("full", "block1", "block2"):
            for left, right in itertools.combinations(tasks, 2):
                rows.append(_comparison_row(
                    "independent_vs_independent",
                    state_name,
                    component,
                    left,
                    right,
                    independent_parts[left][component],
                    independent_parts[right][component],
                ))

        for left, right in itertools.combinations(tasks, 2):
            rows.append(_comparison_row(
                "private_vs_private",
                state_name,
                "private",
                left,
                right,
                shared_parts[left]["private"],
                shared_parts[right]["private"],
            ))

        shared_prefix = shared_parts[reference_task]["shared"]
        for task in tasks:
            rows.append(_comparison_row(
                "shared_vs_private",
                state_name,
                "private",
                "shared",
                task,
                shared_prefix,
                shared_parts[task]["private"],
            ))
            for component in ("block1", "block2"):
                rows.append(_comparison_row(
                    "shared_vs_independent",
                    state_name,
                    component,
                    "shared",
                    task,
                    shared_prefix,
                    independent_parts[task][component],
                ))

        for component in ("block1", "block2"):
            task_mean = torch.stack(
                [independent_parts[task][component] for task in tasks],
                dim=0,
            ).mean(dim=0)
            rows.append(_comparison_row(
                "shared_vs_independent_mean",
                state_name,
                component,
                "shared",
                "task_mean",
                shared_prefix,
                task_mean,
            ))
    return rows


def analyze_artifacts(
    independent: dict[str, Any],
    shared: dict[str, Any],
    tasks: list[str],
) -> dict[str, Any]:
    if int(independent["prefix_length"]) != int(shared["prefix_length"]):
        raise ValueError("independent/shared prefix lengths do not match")
    if int(independent["hidden_size"]) != int(shared["hidden_size"]):
        raise ValueError("independent/shared hidden sizes do not match")
    decomposition = (
        build_decomposition_rows("independent", independent, tasks)
        + build_decomposition_rows("shared_task", shared, tasks)
    )
    transitions = (
        build_transition_rows("independent", independent, tasks)
        + build_transition_rows("shared_task", shared, tasks)
    )
    structures = build_structure_rows(independent, shared, tasks)
    return {
        "decomposition": decomposition,
        "decomposition_aggregates": aggregate_decomposition_rows(decomposition),
        "transitions": transitions,
        "transition_aggregates": aggregate_rows(transitions),
        "structures": structures,
        "structure_aggregates": aggregate_rows(structures),
    }


def _fake_artifact(layout: str) -> dict[str, Any]:
    tasks = ["a", "b", "c"]
    initial: dict[str, torch.Tensor] = {}
    final: dict[str, torch.Tensor] = {}
    residual: dict[str, torch.Tensor] = {}
    folded: dict[str, torch.Tensor] = {}
    torch.manual_seed(11 if layout == "independent_2x16" else 13)
    shared_block = torch.randn(2, 5)
    for index, task in enumerate(tasks):
        if layout == "independent_2x16":
            value = torch.randn(4, 5)
        else:
            value = torch.cat([shared_block, torch.randn(2, 5)], dim=0)
        initial[task] = value
        final[task] = value + 0.01 * (index + 1)
        if layout == "shared16_task16":
            final[task][:2] = initial[task][:2] + 0.01
        residual[task] = torch.full_like(value, 0.02)
        folded[task] = final[task] + residual[task]
    return {
        "schema_version": 1,
        "layout": layout,
        "prefix_length": 2,
        "effective_prefix_length": 4,
        "hidden_size": 5,
        "task_order": tasks,
        "states": {
            "initial": initial,
            "final_raw": final,
            "residual_contribution": residual,
            "folded": folded,
        },
    }


def run_self_test() -> None:
    result = analyze_artifacts(
        _fake_artifact("independent_2x16"),
        _fake_artifact("shared16_task16"),
        ["a", "b", "c"],
    )
    assert result["decomposition"]
    assert result["transitions"]
    assert result["structures"]
    assert max(
        row["folded_reconstruction_max_abs_error"]
        for row in result["decomposition"]
    ) < 1e-6
    print("PREFIX_STATE_TRAJECTORY_SELF_TEST_OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--independent-states",
        default="outputs/v2/four_cell/B_joint_2x16_residual/seed1/prefix_states.pt",
    )
    parser.add_argument(
        "--shared-states",
        default="outputs/v2/four_cell/D_shared16_task16_residual/seed1/prefix_states.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v2/prefix_state_trajectory/B_vs_D_seed1",
    )
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    independent = _load_artifact(
        Path(args.independent_states),
        "independent_2x16",
    )
    shared = _load_artifact(
        Path(args.shared_states),
        "shared16_task16",
    )
    tasks = _common_tasks(independent, shared, list(args.tasks))
    result = analyze_artifacts(independent, shared, tasks)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "decomposition_csv": output_dir / "prefix_state_decomposition.csv",
        "decomposition_aggregates_csv": output_dir / "prefix_state_decomposition_aggregates.csv",
        "transitions_csv": output_dir / "prefix_state_transitions.csv",
        "transition_aggregates_csv": output_dir / "prefix_state_transition_aggregates.csv",
        "structure_csv": output_dir / "prefix_state_structure.csv",
        "structure_aggregates_csv": output_dir / "prefix_state_structure_aggregates.csv",
        "json": output_dir / "prefix_state_analysis.json",
    }
    _write_csv(paths["decomposition_csv"], result["decomposition"])
    _write_csv(
        paths["decomposition_aggregates_csv"],
        result["decomposition_aggregates"],
    )
    _write_csv(paths["transitions_csv"], result["transitions"])
    _write_csv(paths["transition_aggregates_csv"], result["transition_aggregates"])
    _write_csv(paths["structure_csv"], result["structures"])
    _write_csv(paths["structure_aggregates_csv"], result["structure_aggregates"])
    payload = {
        "schema_version": 1,
        "independent_states": independent["_path"],
        "shared_states": shared["_path"],
        "tasks": tasks,
        "interpretation": {
            "initial_to_final_raw": "change attributable to joint task training",
            "final_raw_to_folded": "change attributable to the trained residual MLP",
            "initial_to_folded": "combined end-to-end change",
            "residual_contribution": "folded - final_raw, saved exactly",
        },
        **result,
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "ok",
        "tasks": tasks,
        "decomposition_rows": len(result["decomposition"]),
        "transition_rows": len(result["transitions"]),
        "structure_rows": len(result["structures"]),
        "output_dir": str(output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
