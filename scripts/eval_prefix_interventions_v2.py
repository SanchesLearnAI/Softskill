#!/usr/bin/env python3
"""Evaluate inference-only functional interventions on a v2-D checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skillopt.evaluation.gate import select_gate_score
from skillopt.softprefix.interventions_v2 import (
    DEFAULT_RESIDUAL_ALPHAS,
    RESTORE_BOTH_INITIAL,
    RESTORE_FINAL,
    RESTORE_SHARED_INITIAL,
    RESTORE_TASK_INITIAL,
    IntervenableJointV2SoftPrefixVisionLM,
    apply_final_residual_with_alpha,
    construct_raw_intervention,
    parse_residual_alphas,
    prefix_diagnostics,
    select_global_alpha,
    validate_shared_task_artifact,
)
from skillopt.softprefix.multitask_trainer import _load_task
from skillopt.softprefix.multitask_v2 import (
    _load_task_markdown,
    initialization_kwargs_from_checkpoint_state,
)
from skillopt.softprefix.trainer import _evaluate_prefix


DEFAULT_CHECKPOINT = "outputs/v2/four_cell/D_shared16_task16_residual/seed1/best_v2.pt"
DEFAULT_PREFIX_STATES = "outputs/v2/four_cell/D_shared16_task16_residual/seed1/prefix_states.pt"
DEFAULT_OUTPUT_ROOT = "outputs/v2/interventions/D_shared16_task16_residual/seed1"
DEFAULT_TASK_SPECS = (
    {
        "name": "searchqa",
        "env": "searchqa",
        "config": "configs/searchqa/soft_prefix_2x16.yaml",
        "split_dir": "data/searchqa_split",
    },
    {
        "name": "livemath",
        "env": "livemathematicianbench",
        "config": "configs/livemathematicianbench/soft_prefix_2x16.yaml",
        "split_dir": "data/livemathematicianbench_split",
    },
    {
        "name": "docvqa",
        "env": "docvqa",
        "config": "configs/docvqa/soft_prefix_2x16.yaml",
        "split_dir": "data/docvqa/splits",
    },
)


def _load_torch_file(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=_PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short").splitlines(),
    }


def _alpha_slug(alpha: float) -> str:
    return format(float(alpha), ".9g").replace("-", "m").replace(".", "p")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _write_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    _write_json(root / "intervention_summary.json", rows)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path = root / "intervention_summary.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)


def _condition_dir(
    stage_root: Path,
    *,
    family: str,
    alpha: float,
    task_name: str,
    condition: str,
    donor_task: str | None,
) -> Path:
    parts = [family, f"alpha_{_alpha_slug(alpha)}", condition]
    if donor_task is not None:
        parts.append(f"donor_{donor_task}")
    parts.append(task_name)
    return stage_root.joinpath(*parts)


def _evaluate_one(
    torch,
    *,
    model,
    task,
    artifact: dict[str, Any],
    stage_root: Path,
    stage: str,
    family: str,
    alpha: float,
    condition: str = RESTORE_FINAL,
    donor_task: str | None = None,
) -> dict[str, Any]:
    model.set_active_task(task.name)
    raw_cpu = construct_raw_intervention(
        artifact,
        target_task=task.name,
        condition=condition,
        donor_task=donor_task,
    )
    raw = raw_cpu.to(device=model.device, dtype=model.prefix_embeddings.dtype)
    with torch.no_grad():
        deployed = apply_final_residual_with_alpha(model, raw, alpha=alpha)
    model.set_prefix_override(deployed)
    out_dir = _condition_dir(
        stage_root,
        family=family,
        alpha=alpha,
        task_name=task.name,
        condition=condition,
        donor_task=donor_task,
    )
    items = task.val_items if stage == "validation" else task.test_items
    try:
        hard, soft, _ = _evaluate_prefix(
            task.env,
            model,
            items,
            cfg=task.cfg,
            settings=task.settings,
            out_dir=str(out_dir),
            desc=f"{stage} {family} {condition} {task.name}",
        )
    finally:
        model.clear_prefix_override()
    gate_metric = str(task.cfg.get("gate_metric", "hard") or "hard")
    mixed_weight = float(task.cfg.get("gate_mixed_weight", 0.5) or 0.5)
    row: dict[str, Any] = {
        "stage": stage,
        "family": family,
        "condition": condition,
        "target_task": task.name,
        "donor_task": donor_task or "",
        "alpha": float(alpha),
        "hard": float(hard),
        "soft": float(soft),
        "score": select_gate_score(hard, soft, gate_metric, mixed_weight),
        "gate_metric": gate_metric,
        "n_items": len(items),
        "results_path": str((out_dir / "results.jsonl").resolve()),
    }
    row.update(prefix_diagnostics(torch, raw, deployed))
    return row


def _reused_swap_row(base: dict[str, Any], *, task_name: str) -> dict[str, Any]:
    row = dict(base)
    row.update({
        "family": "task_swap",
        "condition": RESTORE_FINAL,
        "donor_task": task_name,
        "reused_from": "alpha/full",
    })
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validation", "test"], default="validation")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prefix-states", default=DEFAULT_PREFIX_STATES)
    parser.add_argument("--out-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_RESIDUAL_ALPHAS),
    )
    parser.add_argument("--selection-manifest")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    states_path = Path(args.prefix_states).resolve()
    for path in (checkpoint_path, states_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    out_root = Path(args.out_root).resolve()
    stage_root = out_root / args.stage
    if stage_root.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse {stage_root}; pass --resume")
    stage_root.mkdir(parents=True, exist_ok=True)

    import torch

    checkpoint = _load_torch_file(torch, checkpoint_path)
    state = checkpoint.get("model")
    if not isinstance(state, dict) or state.get("protocol_version") != "v2":
        raise ValueError("checkpoint does not contain a v2 model state")
    artifact = _load_torch_file(torch, states_path)
    task_order = validate_shared_task_artifact(
        artifact,
        checkpoint_tasks=state.get("task_prefix_embeddings", {}),
    )
    task_specs = [dict(spec) for spec in DEFAULT_TASK_SPECS]
    if [spec["name"] for spec in task_specs] != task_order:
        raise ValueError("default task order differs from the prefix-state artifact")
    task_init_texts = _load_task_markdown(task_specs)
    initialization_kwargs = initialization_kwargs_from_checkpoint_state(
        state,
        fallback_task_init_texts=task_init_texts,
    )

    model = IntervenableJointV2SoftPrefixVisionLM(
        args.model_name,
        layout=str(state["layout"]),
        prefix_length=int(state["prefix_length"]),
        **initialization_kwargs,
        residual_bottleneck_size=int(state["residual_bottleneck_size"]),
        use_residual_reparameterization=bool(state["use_residual_reparameterization"]),
        residual_mode=str(state["residual_mode"]),
        residual_init_seed=int(state["residual_init_seed"]),
        torch_dtype=args.torch_dtype,
        device=args.device,
        trust_remote_code=True,
    )
    model.load_state_dict(state)
    tasks = [_load_task(spec, model=model, seed=args.seed) for spec in task_specs]

    with torch.no_grad():
        for task_name in task_order:
            model.set_active_task(task_name)
            raw = construct_raw_intervention(artifact, target_task=task_name).to(
                device=model.device,
                dtype=model.prefix_embeddings.dtype,
            )
            alpha_zero = apply_final_residual_with_alpha(model, raw, alpha=0.0)
            alpha_one = apply_final_residual_with_alpha(model, raw, alpha=1.0)
            if not torch.equal(alpha_zero.cpu(), artifact["states"]["final_raw"][task_name].to(alpha_zero.dtype)):
                raise ValueError(f"alpha=0 invariant failed for {task_name}")
            folded = artifact["states"]["folded"][task_name].to(alpha_one.dtype)
            if not torch.allclose(alpha_one.cpu(), folded, atol=1e-5, rtol=1e-5):
                error = float((alpha_one.cpu() - folded).abs().max().item())
                raise ValueError(f"alpha=1 invariant failed for {task_name}: {error}")

    audit = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "prefix_states": str(states_path),
        "prefix_states_sha256": _sha256(states_path),
        "model_name": args.model_name,
        "seed": args.seed,
        "task_order": task_order,
        "alphas": parse_residual_alphas(args.alphas),
        "selection_rule": "maximum validation macro score; exact ties choose smaller alpha",
        "artifact_invariants": artifact.get("invariants", {}),
        "git": _git_snapshot(),
    }
    _write_json(stage_root / "protocol_audit.json", audit)

    rows: list[dict[str, Any]] = []
    alpha_rows: dict[tuple[str, float], dict[str, Any]] = {}
    if args.stage == "validation":
        alphas = parse_residual_alphas(args.alphas)
    else:
        manifest_path = Path(args.selection_manifest).resolve() if args.selection_manifest else out_root / "selection_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"test stage requires selection manifest: {manifest_path}")
        selection = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected_alpha = float(selection["selected"]["alpha"])
        alphas = [selected_alpha] + ([] if selected_alpha == 1.0 else [1.0])

    for alpha in alphas:
        for task in tasks:
            row = _evaluate_one(
                torch,
                model=model,
                task=task,
                artifact=artifact,
                stage_root=stage_root,
                stage=args.stage,
                family="alpha",
                alpha=alpha,
            )
            rows.append(row)
            alpha_rows[(task.name, float(alpha))] = row
            _write_rows(stage_root, rows)

    if args.stage == "validation":
        selection = select_global_alpha(rows, task_order)
        selected_alpha = float(selection["selected"]["alpha"])
        selection.update({
            "stage": "validation",
            "selection_rule": audit["selection_rule"],
            "checkpoint_sha256": audit["checkpoint_sha256"],
            "prefix_states_sha256": audit["prefix_states_sha256"],
        })
        _write_json(out_root / "selection_manifest.json", selection)

    for condition in (RESTORE_SHARED_INITIAL, RESTORE_TASK_INITIAL, RESTORE_BOTH_INITIAL):
        for task in tasks:
            rows.append(_evaluate_one(
                torch,
                model=model,
                task=task,
                artifact=artifact,
                stage_root=stage_root,
                stage=args.stage,
                family="restore",
                alpha=selected_alpha,
                condition=condition,
            ))
            _write_rows(stage_root, rows)

    for task in tasks:
        base = alpha_rows[(task.name, selected_alpha)]
        rows.append(_reused_swap_row(base, task_name=task.name))
        for donor_task in task_order:
            if donor_task == task.name:
                continue
            rows.append(_evaluate_one(
                torch,
                model=model,
                task=task,
                artifact=artifact,
                stage_root=stage_root,
                stage=args.stage,
                family="task_swap",
                alpha=selected_alpha,
                donor_task=donor_task,
            ))
            _write_rows(stage_root, rows)

    _write_rows(stage_root, rows)
    _write_json(stage_root / "completed.json", {
        "status": "ok",
        "stage": args.stage,
        "selected_alpha": selected_alpha,
        "rows": len(rows),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({
        "status": "ok",
        "stage": args.stage,
        "selected_alpha": selected_alpha,
        "rows": len(rows),
        "output": str(stage_root),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
