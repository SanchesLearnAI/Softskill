"""Process-isolated, resumable ChartQA evaluation for step-150 checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import file_sha256
from skillopt.behavior_compression.new_task_transfer.v1.drop_chunked_eval import (
    _atomic_json,
    _atomic_jsonl,
    _ids_sha256,
    _progress_summary,
    _read_json,
    _read_jsonl,
    _torch_load,
    chunk_ranges,
)
from skillopt.behavior_compression.new_task_transfer.v1.evaluation import (
    _resolve_generation,
    evaluate_transfer_predictions,
)
from skillopt.behavior_compression.new_task_transfer.v1.model import (
    TargetOnlyTransferSoftPrefixVisionLM,
)
from skillopt.envs.chartqa.data import load_chartqa_human_split
from skillopt.envs.chartqa.evaluator import evaluate as evaluate_chartqa


PROTOCOL_VERSION = "chartqa_chunked_eval_v1"
FINAL_OPTIMIZER_STEP = 150
DEFAULT_CHUNK_SIZE = 256
TASK = "chartqa"
VALID_CONDITIONS = {"f0", "f1", "i32"}
VALID_SPLITS = {"validation": "val", "test": "test"}


def chunk_paths(
    output_dir: str | os.PathLike[str],
    split: str,
    start: int,
    end: int,
) -> dict[str, Path]:
    split = str(split).lower()
    if split not in VALID_SPLITS:
        raise ValueError(f"invalid ChartQA split: {split}")
    root = Path(output_dir).expanduser().resolve() / "predictions" / "chunks" / split
    stem = f"{split}_{int(start):06d}_{int(end):06d}"
    return {
        "predictions": root / f"{stem}.jsonl",
        "progress": root / f"{stem}.progress.jsonl",
        "manifest": root / f"{stem}.manifest.json",
    }


def validate_existing_run(
    *,
    torch_module: Any,
    output_dir: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    condition: str,
    seed: int,
    source_checkpoint: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    output = Path(output_dir).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    condition = str(condition).lower()
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"invalid ChartQA condition: {condition}")
    audit = _read_json(output / "step0_audit.json")
    sanity = _read_json(output / "transfer_sanity.json")
    for name, payload in (("audit", audit), ("sanity", sanity)):
        for key, expected in (("task", TASK), ("condition", condition), ("seed", int(seed))):
            if payload.get(key) != expected:
                raise RuntimeError(f"{name} {key} mismatch")
        if payload.get("status") != "passed":
            raise RuntimeError(f"{name} did not pass")
    if int(audit.get("prefix_total_length", -1)) != 32:
        raise RuntimeError("ChartQA checkpoint does not use a 32-token prefix")
    if bool(audit.get("residual_enabled", True)):
        raise RuntimeError("ChartQA checkpoint unexpectedly enables residual")
    if not bool(audit.get("base_model", {}).get("fully_frozen", False)):
        raise RuntimeError("ChartQA base model was not fully frozen")

    if condition == "f1":
        if not source_checkpoint:
            raise RuntimeError("ChartQA F1 requires an explicit source checkpoint")
        source = Path(source_checkpoint).expanduser().resolve()
        recorded = Path(audit.get("f1_source", {}).get("checkpoint_path", "")).expanduser().resolve()
        if source != recorded:
            raise RuntimeError("ChartQA F1 source checkpoint path mismatch")
        if file_sha256(source) != audit.get("f1_source", {}).get("checkpoint_sha256"):
            raise RuntimeError("ChartQA F1 source checkpoint hash mismatch")
    elif source_checkpoint:
        raise RuntimeError("only ChartQA F1 may receive a source checkpoint")

    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"missing ChartQA final checkpoint: {checkpoint_file}")
    checkpoint = _torch_load(torch_module, checkpoint_file)
    for key, expected in (("task", TASK), ("condition", condition), ("seed", int(seed))):
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"ChartQA final checkpoint {key} mismatch")
    if int(checkpoint.get("optimizer_step", -1)) != FINAL_OPTIMIZER_STEP:
        raise RuntimeError("ChartQA final checkpoint is not step 150")
    if checkpoint.get("sample_order_hash") != audit.get("sample_order_hash"):
        raise RuntimeError("ChartQA checkpoint training sample-order hash mismatch")
    if not isinstance(checkpoint.get("prefix_state"), dict):
        raise RuntimeError("ChartQA final checkpoint lacks prefix-only state")
    return audit, checkpoint, file_sha256(checkpoint_file)


def _score_records(
    records: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, float]:
    expected_ids = [str(item["id"]) for item in items]
    actual_ids = [str(record.get("sample_id", "")) for record in records]
    if actual_ids != expected_ids:
        raise RuntimeError("ChartQA chunk predictions are incomplete or out of order")
    values: list[float] = []
    for record, item in zip(records, items):
        evaluated = evaluate_chartqa(str(record.get("raw_prediction", "")), item["answer"])
        score = float(evaluated["relaxed_accuracy"])
        stored = float(record.get("relaxed_accuracy", float("nan")))
        if not math.isfinite(stored) or abs(stored - score) > 1e-12:
            raise RuntimeError(
                f"stored ChartQA score differs from official evaluator for {item['id']}"
            )
        values.append(score)
    return {"relaxed_accuracy": sum(values) / len(values)}


def _resolved_generation(config: dict[str, Any] | None) -> dict[str, Any]:
    generation = _resolve_generation(TASK, config)
    generation["batch_size"] = 1
    generation["message_order"] = "question_then_image"
    return generation


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    predictions_path: Path,
    items: list[dict[str, Any]],
    condition: str,
    split: str,
    start: int,
    end: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    seed: int,
    generation: dict[str, Any],
) -> dict[str, float]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "task": TASK,
        "condition": condition,
        "seed": int(seed),
        "split": split,
        "start": int(start),
        "end": int(end),
        "sample_count": int(end - start),
        "sample_ids_sha256": _ids_sha256(items),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "generation": generation,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"ChartQA chunk manifest {key} mismatch: {predictions_path}")
    if not predictions_path.is_file():
        raise RuntimeError(f"ChartQA chunk manifest exists without predictions: {predictions_path}")
    if manifest.get("predictions_sha256") != file_sha256(predictions_path):
        raise RuntimeError(f"ChartQA chunk prediction hash mismatch: {predictions_path}")
    scores = _score_records(_read_jsonl(predictions_path), items)
    stored = float(manifest.get("metrics", {}).get("relaxed_accuracy", float("nan")))
    if not math.isfinite(stored) or abs(stored - scores["relaxed_accuracy"]) > 1e-12:
        raise RuntimeError(f"ChartQA chunk manifest metrics mismatch: {predictions_path}")
    return scores


def run_chartqa_chunk(
    *,
    model: Any,
    dataset: list[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    checkpoint_sha256: str,
    condition: str,
    seed: int,
    split: str,
    start: int,
    chunk_size: int,
    generation_config: dict[str, Any] | None = None,
    evaluator: Callable[..., dict[str, Any]] = evaluate_transfer_predictions,
) -> dict[str, Any]:
    condition = str(condition).lower()
    split = str(split).lower()
    ranges = dict(chunk_ranges(len(dataset), chunk_size))
    start = int(start)
    if start not in ranges:
        raise ValueError(f"chunk start {start} is not aligned to chunk_size={chunk_size}")
    end = ranges[start]
    items = dataset[start:end]
    paths = chunk_paths(output_dir, split, start, end)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    generation = _resolved_generation(generation_config)

    if paths["manifest"].is_file():
        manifest = _read_json(paths["manifest"])
        _validate_manifest(
            manifest=manifest,
            predictions_path=paths["predictions"],
            items=items,
            condition=condition,
            split=split,
            start=start,
            end=end,
            checkpoint_path=checkpoint_file,
            checkpoint_sha256=checkpoint_sha256,
            seed=seed,
            generation=generation,
        )
        result = dict(manifest)
        result["reused"] = True
        return result

    started = time.perf_counter()
    if paths["predictions"].is_file():
        metrics = _score_records(_read_jsonl(paths["predictions"]), items)
        evaluation_elapsed = 0.0
        recovered_predictions = True
    else:
        result = evaluator(
            model=model,
            task=TASK,
            dataset=items,
            split=split,
            batch_size=1,
            generation_config=generation_config,
            predictions_path=paths["predictions"],
            resume_partial=True,
            progress_path=paths["progress"],
            checkpoint_interval=25,
        )
        if not bool(result.get("complete", False)) or not paths["predictions"].is_file():
            raise RuntimeError("ChartQA chunk evaluator did not publish complete predictions")
        metrics = _score_records(_read_jsonl(paths["predictions"]), items)
        evaluation_elapsed = time.perf_counter() - started
        recovered_predictions = False

    progress = _progress_summary(paths["progress"], [str(item["id"]) for item in items])
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "task": TASK,
        "condition": condition,
        "seed": int(seed),
        "split": split,
        "start": start,
        "end": end,
        "sample_count": len(items),
        "sample_ids_sha256": _ids_sha256(items),
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_sha256": checkpoint_sha256,
        "generation": generation,
        "metrics": metrics,
        "evaluation_process_seconds": evaluation_elapsed,
        "progress": progress,
        "predictions_path": str(paths["predictions"]),
        "predictions_sha256": file_sha256(paths["predictions"]),
        "recovered_complete_predictions": recovered_predictions,
    }
    _atomic_json(paths["manifest"], manifest)
    manifest["reused"] = False
    return manifest


def _upsert_eval_record(path: Path, record: dict[str, Any]) -> None:
    records = _read_jsonl(path) if path.is_file() else []
    if len({str(item.get("split")) for item in records}) != len(records):
        raise RuntimeError("ChartQA eval ledger contains duplicate splits")
    existing = next((item for item in records if item.get("split") == record["split"]), None)
    if existing is not None:
        if existing != record:
            raise RuntimeError(f"existing ChartQA {record['split']} metric conflicts with merge")
        return
    records.append(record)
    expected_order = {"validation": 0, "test": 1}
    records.sort(key=lambda item: expected_order[str(item["split"])])
    _atomic_jsonl(path, records)


def merge_chartqa_chunks(
    *,
    dataset: list[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    checkpoint_sha256: str,
    condition: str,
    seed: int,
    split: str,
    chunk_size: int,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    condition = str(condition).lower()
    split = str(split).lower()
    generation = _resolved_generation(generation_config)
    all_records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for start, end in chunk_ranges(len(dataset), chunk_size):
        paths = chunk_paths(output, split, start, end)
        if not paths["manifest"].is_file():
            raise RuntimeError(f"missing complete ChartQA chunk manifest: {paths['manifest']}")
        manifest = _read_json(paths["manifest"])
        _validate_manifest(
            manifest=manifest,
            predictions_path=paths["predictions"],
            items=dataset[start:end],
            condition=condition,
            split=split,
            start=start,
            end=end,
            checkpoint_path=checkpoint_file,
            checkpoint_sha256=checkpoint_sha256,
            seed=seed,
            generation=generation,
        )
        manifests.append(manifest)
        all_records.extend(_read_jsonl(paths["predictions"]))

    metrics = _score_records(all_records, dataset)
    predictions_path = output / "predictions" / f"{split}.jsonl"
    if predictions_path.is_file():
        if _read_jsonl(predictions_path) != all_records:
            raise RuntimeError(f"existing formal ChartQA {split} predictions conflict with merge")
    else:
        _atomic_jsonl(predictions_path, all_records)

    total_sample_seconds = sum(
        float(manifest.get("progress", {}).get("sample_elapsed_seconds", 0.0))
        for manifest in manifests
    )
    total_process_seconds = sum(
        float(manifest.get("evaluation_process_seconds", 0.0)) for manifest in manifests
    )
    consolidated = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "task": TASK,
        "condition": condition,
        "seed": int(seed),
        "split": split,
        "sample_count": len(dataset),
        "sample_ids_sha256": _ids_sha256(dataset),
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_sha256": checkpoint_sha256,
        "chunk_size": int(chunk_size),
        "chunk_count": len(manifests),
        "generation": generation,
        "metrics": metrics,
        "timing_seconds": {
            "sample_generation": total_sample_seconds,
            "evaluation_processes": total_process_seconds,
        },
        "predictions_path": str(predictions_path),
        "predictions_sha256": file_sha256(predictions_path),
    }
    _atomic_json(output / "predictions" / f"{split}.chunked_manifest.json", consolidated)
    _upsert_eval_record(
        output / "eval_metrics.jsonl",
        {
            "task": TASK,
            "condition": condition,
            "seed": int(seed),
            "checkpoint": "final.pt",
            "optimizer_step": FINAL_OPTIMIZER_STEP,
            "split": split,
            "report_split": split == "test",
            "num_samples": len(dataset),
            "generation": generation,
            "metrics": metrics,
            "elapsed_seconds": total_sample_seconds,
            "recovered_from_complete_predictions": False,
            "execution_mode": "chunked_multi_process",
            "chunk_size": int(chunk_size),
            "chunk_count": len(manifests),
        },
    )
    return consolidated


def _generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": 0.0,
        "max_image_tokens": int(args.max_image_tokens),
        "enable_thinking": False,
        "stop_strings": ["</answer>"],
        "use_cache": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("worker", "merge"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--data_root", required=True)
        sub.add_argument("--output_dir", required=True)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--condition", required=True, choices=tuple(sorted(VALID_CONDITIONS)))
        sub.add_argument("--source_checkpoint", default="")
        sub.add_argument("--seed", type=int, default=1)
        sub.add_argument("--split", required=True, choices=tuple(VALID_SPLITS))
        sub.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
        sub.add_argument("--max_prompt_tokens", type=int, default=8192)
        sub.add_argument("--max_new_tokens", type=int, default=64)
        sub.add_argument("--max_image_tokens", type=int, default=0)
    worker = subparsers.choices["worker"]
    worker.add_argument("--model_name", required=True)
    worker.add_argument("--start", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    import torch

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    _audit, checkpoint, checkpoint_hash = validate_existing_run(
        torch_module=torch,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        condition=args.condition,
        seed=args.seed,
        source_checkpoint=args.source_checkpoint or None,
    )
    dataset = load_chartqa_human_split(args.data_root, VALID_SPLITS[args.split])
    generation_config = _generation_config(args)
    if args.command == "merge":
        result = merge_chartqa_chunks(
            dataset=dataset,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_hash,
            condition=args.condition,
            seed=args.seed,
            split=args.split,
            chunk_size=args.chunk_size,
            generation_config=generation_config,
        )
    else:
        start = int(args.start)
        ranges = dict(chunk_ranges(len(dataset), args.chunk_size))
        if start not in ranges:
            raise ValueError(f"invalid ChartQA chunk start {start}")
        paths = chunk_paths(output_dir, args.split, start, ranges[start])
        if paths["manifest"].is_file():
            model = None
        else:
            model = TargetOnlyTransferSoftPrefixVisionLM(
                args.model_name,
                task_name=TASK,
                condition=args.condition,
                source_checkpoint_path=args.source_checkpoint or None,
                torch_dtype="auto",
                device="auto",
                trust_remote_code=True,
            )
            model.load_state_dict(checkpoint["prefix_state"])
            if int(model.active_prefix_embeddings().shape[0]) != 32:
                raise RuntimeError("loaded ChartQA prefix length is not 32")
            if any(parameter.requires_grad for parameter in model.model.parameters()):
                raise RuntimeError("loaded ChartQA base model is not frozen")
            if bool(getattr(model, "use_residual_reparameterization", False)):
                raise RuntimeError("loaded ChartQA model unexpectedly enables residual")
        result = run_chartqa_chunk(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_hash,
            condition=args.condition,
            seed=args.seed,
            split=args.split,
            start=start,
            chunk_size=args.chunk_size,
            generation_config=generation_config,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
