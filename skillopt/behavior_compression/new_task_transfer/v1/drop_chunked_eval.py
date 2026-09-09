"""Process-isolated, resumable DROP evaluation for an existing final checkpoint.

Each worker evaluates one deterministic validation slice and exits.  Completed
slice files are immutable and can be reused by a later Slurm run.  The merge
command publishes the formal validation predictions and eval ledger only after
all slices pass identity, ordering, checkpoint, and metric validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import (
    file_sha256,
)
from skillopt.behavior_compression.new_task_transfer.v1.evaluation import (
    _resolve_generation,
    evaluate_transfer_predictions,
)
from skillopt.behavior_compression.new_task_transfer.v1.model import (
    TargetOnlyTransferSoftPrefixVisionLM,
)
from skillopt.envs.drop.data import load_drop_split
from skillopt.envs.drop.evaluator import evaluate as evaluate_drop


PROTOCOL_VERSION = "drop_chunked_eval_v1"
FINAL_OPTIMIZER_STEP = 150
DEFAULT_CHUNK_SIZE = 512
TASK = "drop"
SPLIT = "validation"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL line {line_number} is not an object: {path}")
        records.append(value)
    return records


def _ids_sha256(items: list[dict[str, Any]]) -> str:
    encoded = "\n".join(str(item["id"]) for item in items).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def chunk_ranges(total_samples: int, chunk_size: int) -> list[tuple[int, int]]:
    total_samples = int(total_samples)
    chunk_size = int(chunk_size)
    if total_samples < 1:
        raise ValueError("total_samples must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return [
        (start, min(start + chunk_size, total_samples))
        for start in range(0, total_samples, chunk_size)
    ]


def _chunk_stem(start: int, end: int) -> str:
    return f"validation_{int(start):06d}_{int(end):06d}"


def chunk_paths(output_dir: str | os.PathLike[str], start: int, end: int) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve() / "predictions" / "chunks"
    stem = _chunk_stem(start, end)
    return {
        "predictions": root / f"{stem}.jsonl",
        "progress": root / f"{stem}.progress.jsonl",
        "manifest": root / f"{stem}.manifest.json",
    }


def _torch_load(torch_module: Any, path: Path) -> dict[str, Any]:
    try:
        value = torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch_module.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint must be a dictionary: {path}")
    return value


def validate_existing_run(
    *,
    torch_module: Any,
    output_dir: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    condition: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    output = Path(output_dir).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if str(condition).lower() != "i32":
        raise ValueError("the bounded recovery entry supports DROP I32 only")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"missing final checkpoint: {checkpoint_file}")
    audit = _read_json(output / "step0_audit.json")
    sanity = _read_json(output / "transfer_sanity.json")
    for name, payload in (("audit", audit), ("sanity", sanity)):
        for key, expected in (("task", TASK), ("condition", "i32"), ("seed", int(seed))):
            if payload.get(key) != expected:
                raise RuntimeError(f"{name} {key} mismatch")
        if payload.get("status") != "passed":
            raise RuntimeError(f"{name} did not pass")
    if int(audit.get("prefix_total_length", -1)) != 32:
        raise RuntimeError("DROP I32 checkpoint does not use a 32-token prefix")
    if bool(audit.get("residual_enabled", True)):
        raise RuntimeError("DROP I32 checkpoint unexpectedly enables residual")
    if not bool(audit.get("base_model", {}).get("fully_frozen", False)):
        raise RuntimeError("DROP I32 base model was not fully frozen")

    checkpoint = _torch_load(torch_module, checkpoint_file)
    for key, expected in (("task", TASK), ("condition", "i32"), ("seed", int(seed))):
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"final checkpoint {key} mismatch")
    if int(checkpoint.get("optimizer_step", -1)) != FINAL_OPTIMIZER_STEP:
        raise RuntimeError("DROP I32 final checkpoint is not step 150")
    if checkpoint.get("sample_order_hash") != audit.get("sample_order_hash"):
        raise RuntimeError("final checkpoint training sample-order hash mismatch")
    if not isinstance(checkpoint.get("prefix_state"), dict):
        raise RuntimeError("final checkpoint lacks prefix-only state")
    return audit, checkpoint, file_sha256(checkpoint_file)


def _score_records(
    records: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, float]:
    expected_ids = [str(item["id"]) for item in items]
    actual_ids = [str(record.get("sample_id", "")) for record in records]
    if actual_ids != expected_ids:
        raise RuntimeError("DROP chunk predictions are incomplete or out of order")
    em_values: list[float] = []
    f1_values: list[float] = []
    for record, item in zip(records, items):
        evaluated = evaluate_drop(str(record.get("raw_prediction", "")), item["candidate_answers"])
        em = float(evaluated["em"])
        f1 = float(evaluated["f1"])
        stored_em = float(record.get("em", float("nan")))
        stored_f1 = float(record.get("f1", float("nan")))
        if not math.isfinite(stored_em) or abs(stored_em - em) > 1e-12:
            raise RuntimeError(f"stored DROP EM differs from official evaluator for {item['id']}")
        if not math.isfinite(stored_f1) or abs(stored_f1 - f1) > 1e-12:
            raise RuntimeError(f"stored DROP F1 differs from official evaluator for {item['id']}")
        em_values.append(em)
        f1_values.append(f1)
    return {
        "em": sum(em_values) / len(em_values),
        "f1": sum(f1_values) / len(f1_values),
    }


def _progress_summary(path: Path, expected_ids: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {
            "recorded_samples": 0,
            "sample_elapsed_seconds": 0.0,
            "mean_sample_seconds": None,
            "peak_vmrss_kb": None,
            "peak_vmhwm_kb": None,
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
        }
    # DROP contains one repeated query_id in the official validation release.
    # Progress identity therefore uses the evaluator's chunk-local sample_index,
    # while still checking the ID at that exact position.  A dictionary keyed
    # only by sample_id would silently collapse the repeated example.
    by_index: dict[int, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        try:
            sample_index = int(record.get("sample_index", -1))
        except (TypeError, ValueError):
            continue
        sample_id = str(record.get("sample_id", ""))
        if 0 <= sample_index < len(expected_ids) and sample_id == expected_ids[sample_index]:
            by_index[sample_index] = record
    ordered = [by_index[index] for index in range(len(expected_ids)) if index in by_index]
    elapsed = sum(float(record.get("elapsed_seconds", 0.0)) for record in ordered)

    def peak(name: str) -> int | None:
        values = [
            int(record.get("resources", {}).get(name))
            for record in ordered
            if record.get("resources", {}).get(name) is not None
        ]
        return max(values) if values else None

    return {
        "recorded_samples": len(ordered),
        "sample_elapsed_seconds": elapsed,
        "mean_sample_seconds": elapsed / len(ordered) if ordered else None,
        "peak_vmrss_kb": peak("vmrss_kb"),
        "peak_vmhwm_kb": peak("vmhwm_kb"),
        "peak_cuda_allocated_bytes": peak("cuda_allocated_bytes"),
        "peak_cuda_reserved_bytes": peak("cuda_reserved_bytes"),
    }


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    predictions_path: Path,
    items: list[dict[str, Any]],
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
        "condition": "i32",
        "seed": int(seed),
        "split": SPLIT,
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
            raise RuntimeError(f"DROP chunk manifest {key} mismatch: {predictions_path}")
    if not predictions_path.is_file():
        raise RuntimeError(f"chunk manifest exists without predictions: {predictions_path}")
    if manifest.get("predictions_sha256") != file_sha256(predictions_path):
        raise RuntimeError(f"DROP chunk prediction hash mismatch: {predictions_path}")
    scores = _score_records(_read_jsonl(predictions_path), items)
    for key, value in scores.items():
        stored = float(manifest.get("metrics", {}).get(key, float("nan")))
        if not math.isfinite(stored) or abs(stored - value) > 1e-12:
            raise RuntimeError(f"DROP chunk manifest metrics mismatch: {predictions_path}")
    return scores


def run_drop_chunk(
    *,
    model: Any,
    dataset: list[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    checkpoint_sha256: str,
    seed: int,
    start: int,
    chunk_size: int,
    generation_config: dict[str, Any] | None = None,
    evaluator: Callable[..., dict[str, Any]] = evaluate_transfer_predictions,
) -> dict[str, Any]:
    ranges = dict(chunk_ranges(len(dataset), chunk_size))
    start = int(start)
    if start not in ranges:
        raise ValueError(f"chunk start {start} is not aligned to chunk_size={chunk_size}")
    end = ranges[start]
    items = dataset[start:end]
    paths = chunk_paths(output_dir, start, end)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    generation = _resolve_generation(TASK, generation_config)
    generation["batch_size"] = 1
    generation["message_order"] = "passage_then_question"

    if paths["manifest"].is_file():
        manifest = _read_json(paths["manifest"])
        _validate_manifest(
            manifest=manifest,
            predictions_path=paths["predictions"],
            items=items,
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
        records = _read_jsonl(paths["predictions"])
        metrics = _score_records(records, items)
        evaluation_elapsed = 0.0
        recovered_predictions = True
    else:
        result = evaluator(
            model=model,
            task=TASK,
            dataset=items,
            split=SPLIT,
            batch_size=1,
            generation_config=generation_config,
            predictions_path=paths["predictions"],
            resume_partial=True,
            progress_path=paths["progress"],
            checkpoint_interval=25,
        )
        if not bool(result.get("complete", False)) or not paths["predictions"].is_file():
            raise RuntimeError("DROP chunk evaluator did not publish complete predictions")
        metrics = _score_records(_read_jsonl(paths["predictions"]), items)
        evaluation_elapsed = time.perf_counter() - started
        recovered_predictions = False

    progress = _progress_summary(paths["progress"], [str(item["id"]) for item in items])
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "task": TASK,
        "condition": "i32",
        "seed": int(seed),
        "split": SPLIT,
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


def merge_drop_chunks(
    *,
    dataset: list[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    checkpoint_sha256: str,
    seed: int,
    chunk_size: int,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    generation = _resolve_generation(TASK, generation_config)
    generation["batch_size"] = 1
    generation["message_order"] = "passage_then_question"
    all_records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for start, end in chunk_ranges(len(dataset), chunk_size):
        paths = chunk_paths(output, start, end)
        if not paths["manifest"].is_file():
            raise RuntimeError(f"missing complete DROP chunk manifest: {paths['manifest']}")
        manifest = _read_json(paths["manifest"])
        _validate_manifest(
            manifest=manifest,
            predictions_path=paths["predictions"],
            items=dataset[start:end],
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
    predictions_path = output / "predictions" / "validation.jsonl"
    if predictions_path.is_file():
        existing = _read_jsonl(predictions_path)
        if existing != all_records:
            raise RuntimeError("existing formal DROP predictions conflict with chunk merge")
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
        "condition": "i32",
        "seed": int(seed),
        "split": SPLIT,
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
        "chunk_manifests": [str(chunk_paths(output, start, end)["manifest"]) for start, end in chunk_ranges(len(dataset), chunk_size)],
    }
    _atomic_json(output / "predictions" / "validation.chunked_manifest.json", consolidated)

    eval_record = {
        "task": TASK,
        "condition": "i32",
        "seed": int(seed),
        "checkpoint": "final.pt",
        "optimizer_step": FINAL_OPTIMIZER_STEP,
        "split": SPLIT,
        "report_split": True,
        "num_samples": len(dataset),
        "generation": generation,
        "metrics": metrics,
        "elapsed_seconds": total_sample_seconds,
        "recovered_from_complete_predictions": False,
        "execution_mode": "chunked_multi_process",
        "chunk_size": int(chunk_size),
        "chunk_count": len(manifests),
    }
    eval_path = output / "eval_metrics.jsonl"
    if eval_path.is_file():
        existing = _read_jsonl(eval_path)
        if existing != [eval_record]:
            raise RuntimeError("existing eval_metrics.jsonl conflicts with chunk merge")
    else:
        _atomic_jsonl(eval_path, [eval_record])
    return consolidated


def _generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": 0.0,
        "max_image_tokens": 0,
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
        sub.add_argument("--condition", default="i32", choices=("i32",))
        sub.add_argument("--seed", type=int, default=1)
        sub.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
        sub.add_argument("--max_prompt_tokens", type=int, default=8192)
        sub.add_argument("--max_new_tokens", type=int, default=128)
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
    )
    dataset = load_drop_split(args.data_root, SPLIT)
    generation_config = _generation_config(args)

    if args.command == "merge":
        result = merge_drop_chunks(
            dataset=dataset,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_hash,
            seed=args.seed,
            chunk_size=args.chunk_size,
            generation_config=generation_config,
        )
    else:
        start = int(args.start)
        ranges = dict(chunk_ranges(len(dataset), args.chunk_size))
        if start not in ranges:
            raise ValueError(f"invalid chunk start {start}")
        paths = chunk_paths(output_dir, start, ranges[start])
        if paths["manifest"].is_file():
            result = run_drop_chunk(
                model=None,
                dataset=dataset,
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
                seed=args.seed,
                start=start,
                chunk_size=args.chunk_size,
                generation_config=generation_config,
            )
        else:
            model = TargetOnlyTransferSoftPrefixVisionLM(
                args.model_name,
                task_name=TASK,
                condition="i32",
                source_checkpoint_path=None,
                torch_dtype="auto",
                device="auto",
                trust_remote_code=True,
            )
            model.load_state_dict(checkpoint["prefix_state"])
            if int(model.active_prefix_embeddings().shape[0]) != 32:
                raise RuntimeError("loaded DROP I32 prefix length is not 32")
            if any(parameter.requires_grad for parameter in model.model.parameters()):
                raise RuntimeError("loaded DROP I32 base model is not frozen")
            if bool(getattr(model, "use_residual_reparameterization", False)):
                raise RuntimeError("loaded DROP I32 model unexpectedly enables residual")
            result = run_drop_chunk(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
                seed=args.seed,
                start=start,
                chunk_size=args.chunk_size,
                generation_config=generation_config,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
