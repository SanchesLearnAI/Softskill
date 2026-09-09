#!/usr/bin/env python3
"""Compare learned shared and independent soft-prefix structures on CPU.

The script only loads the small v2 checkpoint/folded-prefix artifacts.  It
does not load the frozen language model and does not require a GPU.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


METRIC_NAMES = (
    "cosine_flat",
    "cosine_centered",
    "cosine_position_mean",
    "cosine_position_std",
    "linear_cka",
    "gram_cosine",
    "token_set_cosine",
    "relative_frobenius",
    "norm_ratio",
)


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _matrix(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float().cpu()
    if value.ndim == 3:
        value = value.flatten(0, 1)
    if value.ndim != 2:
        raise ValueError(f"expected [tokens, hidden] tensor, got {tuple(value.shape)}")
    return value.contiguous()


def _cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    denominator = x.norm() * y.norm()
    if float(denominator) <= eps:
        return 0.0
    return float(torch.dot(x.reshape(-1), y.reshape(-1)) / denominator)


def similarity_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Return complementary position-aligned and permutation-tolerant metrics."""
    x = _matrix(x)
    y = _matrix(y)
    if tuple(x.shape) != tuple(y.shape):
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    eps = 1e-12

    cosine_flat = _cosine(x, y)
    cosine_centered = _cosine(x - x.mean(), y - y.mean())

    x_row_norm = x.norm(dim=1).clamp_min(eps)
    y_row_norm = y.norm(dim=1).clamp_min(eps)
    position_cosine = (x * y).sum(dim=1) / (x_row_norm * y_row_norm)

    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    x_gram_centered = x_centered @ x_centered.T
    y_gram_centered = y_centered @ y_centered.T
    linear_cka = _cosine(x_gram_centered, y_gram_centered)

    x_gram = x @ x.T
    y_gram = y @ y.T
    gram_cosine = _cosine(x_gram, y_gram)

    x_unit = x / x_row_norm.unsqueeze(1)
    y_unit = y / y_row_norm.unsqueeze(1)
    token_cosine = x_unit @ y_unit.T
    token_set_cosine = float(
        0.5 * (token_cosine.max(dim=1).values.mean() + token_cosine.max(dim=0).values.mean())
    )

    x_norm = float(x.norm())
    y_norm = float(y.norm())
    relative_frobenius = float((x - y).norm()) / max(0.5 * (x_norm + y_norm), eps)
    norm_ratio = min(x_norm, y_norm) / max(x_norm, y_norm, eps)
    return {
        "cosine_flat": cosine_flat,
        "cosine_centered": cosine_centered,
        "cosine_position_mean": float(position_cosine.mean()),
        "cosine_position_std": float(position_cosine.std(unbiased=False)),
        "linear_cka": linear_cka,
        "gram_cosine": gram_cosine,
        "token_set_cosine": token_set_cosine,
        "relative_frobenius": relative_frobenius,
        "norm_ratio": norm_ratio,
    }


def _load_run(root: Path, checkpoint_name: str, folded_name: str) -> dict[str, Any]:
    checkpoint_path = root / checkpoint_name
    folded_path = root / folded_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not folded_path.is_file():
        raise FileNotFoundError(folded_path)
    checkpoint = _safe_torch_load(checkpoint_path)
    model_state = checkpoint["model"]
    folded = _safe_torch_load(folded_path)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return {
        "root": str(root.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "folded_path": str(folded_path.resolve()),
        "model": model_state,
        "folded": folded,
        "summary": summary,
    }


def _task_order(independent: dict[str, Any], shared: dict[str, Any], requested: list[str]) -> list[str]:
    independent_tasks = set(independent["model"]["task_prefix_embeddings"])
    shared_tasks = set(shared["model"]["task_prefix_embeddings"])
    folded_tasks = set(independent["folded"]) & set(shared["folded"])
    available = independent_tasks & shared_tasks & folded_tasks
    tasks = requested or ["searchqa", "livemath", "docvqa"]
    tasks = [task for task in tasks if task in available]
    if len(tasks) < 2:
        raise ValueError(f"need at least two common tasks, found {sorted(available)}")
    return tasks


def _extract_structures(
    independent: dict[str, Any],
    shared: dict[str, Any],
    tasks: list[str],
    prefix_length: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    independent_raw_full = {
        task: _matrix(independent["model"]["task_prefix_embeddings"][task]) for task in tasks
    }
    independent_folded_full = {task: _matrix(independent["folded"][task]) for task in tasks}
    for task in tasks:
        expected = 2 * prefix_length
        if independent_raw_full[task].shape[0] != expected:
            raise ValueError(f"{task}: expected {expected} independent raw tokens")
        if independent_folded_full[task].shape[0] != expected:
            raise ValueError(f"{task}: expected {expected} independent folded tokens")

    shared_raw = _matrix(shared["model"]["shared_prefix_embeddings"])
    if shared_raw.shape[0] != prefix_length:
        raise ValueError("unexpected raw shared-prefix length")
    shared_private_raw = {
        task: _matrix(shared["model"]["task_prefix_embeddings"][task]) for task in tasks
    }
    shared_folded_full = {task: _matrix(shared["folded"][task]) for task in tasks}
    shared_folded_candidates = {
        task: shared_folded_full[task][:prefix_length] for task in tasks
    }
    reference_task = tasks[0]
    shared_folded = shared_folded_candidates[reference_task]
    max_shared_difference = max(
        float((shared_folded_candidates[task] - shared_folded).abs().max()) for task in tasks
    )

    structures = {
        "raw": {
            "independent_full": independent_raw_full,
            "independent_block1": {task: value[:prefix_length] for task, value in independent_raw_full.items()},
            "independent_block2": {task: value[prefix_length:] for task, value in independent_raw_full.items()},
            "shared": shared_raw,
            "private": shared_private_raw,
        },
        "folded": {
            "independent_full": independent_folded_full,
            "independent_block1": {task: value[:prefix_length] for task, value in independent_folded_full.items()},
            "independent_block2": {task: value[prefix_length:] for task, value in independent_folded_full.items()},
            "shared": shared_folded,
            "private": {
                task: value[prefix_length:] for task, value in shared_folded_full.items()
            },
        },
    }
    invariants = {
        "folded_shared_max_abs_difference_across_tasks": max_shared_difference,
        "prefix_length": int(prefix_length),
        "effective_prefix_length": int(2 * prefix_length),
        "hidden_size": int(shared_raw.shape[1]),
    }
    return structures, invariants


def _comparison_row(
    comparison: str,
    representation: str,
    component: str,
    left: str,
    right: str,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "comparison": comparison,
        "representation": representation,
        "component": component,
        "left": left,
        "right": right,
        "tokens": int(_matrix(x).shape[0]),
        "hidden_size": int(_matrix(x).shape[1]),
    }
    row.update(similarity_metrics(x, y))
    return row


def build_comparisons(structures: dict[str, Any], tasks: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation, values in structures.items():
        shared_prefix = values["shared"]

        # Shared vs independent: compare against both independent 16-token blocks.
        for block_name in ("independent_block1", "independent_block2"):
            block = values[block_name]
            for task in tasks:
                rows.append(_comparison_row(
                    "shared_vs_independent",
                    representation,
                    block_name,
                    "shared",
                    task,
                    shared_prefix,
                    block[task],
                ))
            mean_block = torch.stack([block[task] for task in tasks], dim=0).mean(dim=0)
            rows.append(_comparison_row(
                "shared_vs_independent_mean",
                representation,
                block_name,
                "shared",
                "task_mean",
                shared_prefix,
                mean_block,
            ))

        # Shared vs the task-private block in the Shared+Task model.
        for task in tasks:
            rows.append(_comparison_row(
                "shared_vs_private",
                representation,
                "private",
                "shared",
                task,
                shared_prefix,
                values["private"][task],
            ))

        # Independent vs independent across tasks.
        for component in ("independent_full", "independent_block1", "independent_block2"):
            component_values = values[component]
            for left, right in itertools.combinations(tasks, 2):
                rows.append(_comparison_row(
                    "independent_vs_independent",
                    representation,
                    component,
                    left,
                    right,
                    component_values[left],
                    component_values[right],
                ))

        # Task-private prefixes in D are also independent across tasks.
        for left, right in itertools.combinations(tasks, 2):
            rows.append(_comparison_row(
                "private_vs_private",
                representation,
                "private",
                left,
                right,
                values["private"][left],
                values["private"][right],
            ))

        # Within each independent task, compare its first and second blocks.
        for task in tasks:
            rows.append(_comparison_row(
                "independent_block1_vs_block2",
                representation,
                "within_task",
                f"{task}:block1",
                f"{task}:block2",
                values["independent_block1"][task],
                values["independent_block2"][task],
            ))
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["comparison"], row["representation"], row["component"])].append(row)
    aggregated = []
    for (comparison, representation, component), values in sorted(groups.items()):
        item: dict[str, Any] = {
            "comparison": comparison,
            "representation": representation,
            "component": component,
            "pair_count": len(values),
        }
        for metric in METRIC_NAMES:
            metric_values = [float(value[metric]) for value in values]
            item[f"{metric}_mean"] = sum(metric_values) / len(metric_values)
            item[f"{metric}_min"] = min(metric_values)
            item[f"{metric}_max"] = max(metric_values)
        aggregated.append(item)
    return aggregated


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    torch.manual_seed(7)
    x = torch.randn(4, 8)
    identical = similarity_metrics(x, x.clone())
    opposite = similarity_metrics(x, -x)
    assert math.isclose(identical["cosine_flat"], 1.0, abs_tol=1e-6)
    assert math.isclose(identical["linear_cka"], 1.0, abs_tol=1e-6)
    assert math.isclose(opposite["cosine_flat"], -1.0, abs_tol=1e-6)
    print("PREFIX_SIMILARITY_SELF_TEST_OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--independent-root",
        default="outputs/v2/four_cell/B_joint_2x16_residual/seed1",
        help="v2-B output directory containing best_v2.pt and folded_prefixes.pt",
    )
    parser.add_argument(
        "--shared-root",
        default="outputs/v2/four_cell/D_shared16_task16_residual/seed1",
        help="v2-D output directory containing best_v2.pt and folded_prefixes.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v2/prefix_similarity/B_vs_D_seed1",
    )
    parser.add_argument("--checkpoint-name", default="best_v2.pt")
    parser.add_argument("--folded-name", default="folded_prefixes.pt")
    parser.add_argument("--prefix-length", type=int, default=16)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    independent = _load_run(Path(args.independent_root), args.checkpoint_name, args.folded_name)
    shared = _load_run(Path(args.shared_root), args.checkpoint_name, args.folded_name)
    if independent["model"].get("layout") != "independent_2x16":
        raise ValueError("independent input is not a v2 independent_2x16 checkpoint")
    if shared["model"].get("layout") != "shared16_task16":
        raise ValueError("shared input is not a v2 shared16_task16 checkpoint")

    tasks = _task_order(independent, shared, list(args.tasks))
    structures, invariants = _extract_structures(
        independent,
        shared,
        tasks,
        int(args.prefix_length),
    )
    rows = build_comparisons(structures, tasks)
    aggregates = aggregate_rows(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "prefix_similarity_pairs.csv"
    aggregate_csv_path = output_dir / "prefix_similarity_aggregates.csv"
    json_path = output_dir / "prefix_similarity.json"
    _write_csv(csv_path, rows)
    _write_csv(aggregate_csv_path, aggregates)
    payload = {
        "schema_version": 1,
        "independent_root": independent["root"],
        "shared_root": shared["root"],
        "tasks": tasks,
        "invariants": invariants,
        "metric_definitions": {
            "cosine_flat": "cosine after flattening all token embeddings",
            "cosine_centered": "flattened cosine after subtracting each tensor's scalar mean",
            "cosine_position_mean": "mean cosine between matching token positions",
            "linear_cka": "linear centered-kernel alignment over token-position Gram matrices",
            "gram_cosine": "cosine between uncentered token-position Gram matrices",
            "token_set_cosine": "symmetric nearest-token cosine; tolerant to token permutation",
            "relative_frobenius": "Frobenius distance divided by mean tensor norm; lower is closer",
            "norm_ratio": "smaller tensor norm divided by larger tensor norm",
        },
        "comparisons": rows,
        "aggregates": aggregates,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "comparison_rows": len(rows),
        "aggregate_rows": len(aggregates),
        "csv": str(csv_path),
        "aggregate_csv": str(aggregate_csv_path),
        "json": str(json_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
