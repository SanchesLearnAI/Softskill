"""Independent prediction recording and official-compatible task evaluation."""
from __future__ import annotations

import gc
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

from skillopt.envs.chartqa.data import build_chartqa_messages
from skillopt.envs.chartqa.evaluator import evaluate as evaluate_chartqa
from skillopt.envs.drop.data import build_drop_prompt
from skillopt.envs.drop.evaluator import evaluate as evaluate_drop


VALID_TASKS = {"chartqa", "drop"}
DEFAULT_MAX_NEW_TOKENS = {"chartqa": 64, "drop": 128}
DEFAULT_MAX_PROMPT_TOKENS = 8192
_GENERATION_KEYS = {
    "max_new_tokens",
    "max_prompt_tokens",
    "temperature",
    "max_image_tokens",
    "enable_thinking",
    "stop_strings",
    "use_cache",
}


def _dataset_items(dataset: Any) -> list[dict[str, Any]]:
    source = dataset.items if hasattr(dataset, "items") else dataset
    if isinstance(source, dict) or not isinstance(source, Iterable):
        raise TypeError("evaluation dataset must be an ordered item sequence")
    items = list(source)
    if not items:
        raise ValueError("evaluation dataset cannot be empty")
    if any(not isinstance(item, dict) for item in items):
        raise TypeError("every evaluation item must be a dictionary")
    return items


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    return {"val": "validation", "dev": "validation"}.get(normalized, normalized)


def _resolve_generation(task: str, supplied: dict[str, Any] | None) -> dict[str, Any]:
    supplied = dict(supplied or {})
    unknown = sorted(set(supplied).difference(_GENERATION_KEYS))
    if unknown:
        raise ValueError(f"unsupported generation parameters: {unknown}")
    if bool(supplied.get("enable_thinking", False)):
        raise ValueError("new-task transfer evaluation requires enable_thinking=False")
    stop_strings = supplied.get("stop_strings", ["</answer>"])
    if isinstance(stop_strings, str):
        stop_strings = [stop_strings]
    if not isinstance(stop_strings, (list, tuple)) or not all(
        isinstance(value, str) and value for value in stop_strings
    ):
        raise ValueError("stop_strings must be a non-empty string sequence")
    use_cache = bool(supplied.get("use_cache", True))
    if not use_cache:
        raise ValueError("new-task transfer evaluation requires use_cache=True")
    resolved = {
        "max_prompt_tokens": int(
            supplied.get("max_prompt_tokens", DEFAULT_MAX_PROMPT_TOKENS)
        ),
        "max_new_tokens": int(
            supplied.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS[task])
        ),
        "temperature": float(supplied.get("temperature", 0.0)),
        "max_image_tokens": int(supplied.get("max_image_tokens", 0)),
        "enable_thinking": False,
        "use_prefix": True,
        "stop_strings": list(stop_strings),
        "use_cache": True,
    }
    if resolved["max_prompt_tokens"] < 1 or resolved["max_new_tokens"] < 1:
        raise ValueError("generation token limits must be positive")
    if resolved["temperature"] < 0 or resolved["max_image_tokens"] < 0:
        raise ValueError("temperature and max_image_tokens cannot be negative")
    return resolved


def _validate_items(task: str, split: str, items: list[dict[str, Any]]) -> None:
    for item in items:
        item_split = _normalize_split(str(item.get("split", split)))
        if item_split != split:
            raise ValueError(
                f"item {item.get('id', '<unknown>')} belongs to {item_split}, not {split}"
            )
        if task == "chartqa":
            if str(item.get("source_type", "")).lower() != "human":
                raise ValueError("ChartQA evaluation accepts only human-authored items")
            for key in ("id", "question", "image_path", "answer"):
                if key not in item:
                    raise ValueError(f"ChartQA evaluation item is missing {key}")
        else:
            for key in (
                "id",
                "passage",
                "question",
                "candidate_answers",
                "answer_type",
            ):
                if key not in item:
                    raise ValueError(f"DROP evaluation item is missing {key}")


def _chunks(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _chartqa_prediction(
    model: Any,
    item: dict[str, Any],
    generation: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    messages = build_chartqa_messages(item)
    raw = model.generate_from_messages(
        messages,
        max_prompt_tokens=generation["max_prompt_tokens"],
        max_new_tokens=generation["max_new_tokens"],
        temperature=generation["temperature"],
        max_image_tokens=generation["max_image_tokens"],
        use_prefix=True,
        stop_strings=generation["stop_strings"],
        use_cache=generation["use_cache"],
    )
    evaluated = evaluate_chartqa(raw, item["answer"])
    record = {
        "sample_id": str(item["id"]),
        "gold_answer": str(item["answer"]),
        "raw_prediction": str(raw),
        "normalized_prediction": evaluated["predicted_answer"],
        "relaxed_accuracy": float(evaluated["relaxed_accuracy"]),
    }
    return record, float(evaluated["relaxed_accuracy"])


def _drop_prediction(
    model: Any,
    item: dict[str, Any],
    generation: dict[str, Any],
) -> tuple[dict[str, Any], float, float]:
    prompt = build_drop_prompt(model.tokenizer, item)
    raw = model.generate_from_prompt(
        prompt,
        max_prompt_tokens=generation["max_prompt_tokens"],
        max_new_tokens=generation["max_new_tokens"],
        temperature=generation["temperature"],
        use_prefix=True,
        stop_strings=generation["stop_strings"],
        use_cache=generation["use_cache"],
    )
    evaluated = evaluate_drop(raw, item["candidate_answers"])
    record = {
        "sample_id": str(item["id"]),
        "gold_answers": evaluated["gold_answers"],
        "raw_prediction": str(raw),
        "parsed_prediction": evaluated["predicted_answer"],
        "answer_type": str(item["answer_type"]),
        "em": float(evaluated["em"]),
        "f1": float(evaluated["f1"]),
    }
    return record, float(evaluated["em"]), float(evaluated["f1"])


def _read_resumable_records(path: Path) -> list[dict[str, Any]]:
    """Read a partial JSONL, dropping only a torn final line after abrupt kill."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    repair = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise RuntimeError(f"malformed non-final line in partial predictions: {path}")
            repair = True
            break
        if not isinstance(value, dict):
            raise RuntimeError(f"partial prediction line is not an object: {path}")
        records.append(value)
    if repair:
        temporary = path.with_suffix(path.suffix + ".repair")
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    return records


def _resource_snapshot(model: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                snapshot[f"{key.lower()}_kb"] = int(value.strip().split()[0])
    torch_module = getattr(model, "torch", None)
    if (
        torch_module is not None
        and hasattr(torch_module, "cuda")
        and torch_module.cuda.is_available()
    ):
        snapshot["cuda_allocated_bytes"] = int(torch_module.cuda.memory_allocated())
        snapshot["cuda_reserved_bytes"] = int(torch_module.cuda.memory_reserved())
        snapshot["cuda_max_allocated_bytes"] = int(
            torch_module.cuda.max_memory_allocated()
        )
    return snapshot


def _prediction_token_count(model: Any, raw_prediction: str) -> int | None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer(str(raw_prediction), add_special_tokens=False)
        return len(encoded["input_ids"])
    except Exception:
        return None


def _scores_from_records(task: str, records: list[dict[str, Any]]):
    if task == "chartqa":
        return [float(record["relaxed_accuracy"]) for record in records], [], []
    return (
        [],
        [float(record["em"]) for record in records],
        [float(record["f1"]) for record in records],
    )


def evaluate_transfer_predictions(
    *,
    model: Any,
    task: str,
    dataset: Any,
    split: str,
    batch_size: int = 1,
    generation_config: dict[str, Any] | None = None,
    predictions_path: str | os.PathLike[str],
    resume_partial: bool = False,
    progress_path: str | os.PathLike[str] | None = None,
    checkpoint_interval: int = 25,
) -> dict[str, Any]:
    """Evaluate and atomically publish JSONL, optionally resuming partial work.

    ``batch_size`` controls deterministic chunking. The current shared model
    wrapper exposes single-example multimodal generation, so examples within a
    chunk are generated serially and always retain source order.
    """
    task = str(task).strip().lower()
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
    split = _normalize_split(split)
    if not split:
        raise ValueError("split is required")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    checkpoint_interval = int(checkpoint_interval)
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    items = _dataset_items(dataset)
    _validate_items(task, split, items)
    generation = _resolve_generation(task, generation_config)

    output = Path(predictions_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"predictions output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    incomplete = output.with_name(
        f".{output.name}.partial" if resume_partial else f".{output.name}.incomplete.{os.getpid()}"
    )
    progress = (
        Path(progress_path).expanduser().resolve()
        if progress_path is not None
        else output.with_name(f".{output.name}.progress.jsonl")
    )
    existing_records = _read_resumable_records(incomplete) if resume_partial else []
    expected_ids = [str(item["id"]) for item in items]
    existing_ids = [str(record.get("sample_id", "")) for record in existing_records]
    if existing_ids != expected_ids[: len(existing_ids)]:
        raise RuntimeError("partial predictions do not match the requested dataset order")
    if len(existing_records) > len(items):
        raise RuntimeError("partial predictions contain more records than the dataset")
    if incomplete.exists():
        if not resume_partial:
            raise FileExistsError(f"incomplete output path already exists: {incomplete}")
    elif existing_records:
        raise RuntimeError("resumable records were read without a partial file")

    chart_scores, drop_em, drop_f1 = _scores_from_records(task, existing_records)
    no_grad = (
        model.torch.no_grad()
        if hasattr(model, "torch") and hasattr(model.torch, "no_grad")
        else nullcontext()
    )
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
    try:
        mode = "a" if incomplete.exists() else "x"
        progress_mode = "a" if progress.exists() and resume_partial else "x"
        with incomplete.open(mode, encoding="utf-8", newline="\n") as handle, progress.open(
            progress_mode, encoding="utf-8", newline="\n"
        ) as progress_handle:
            with no_grad:
                pending = items[len(existing_records) :]
                completed = len(existing_records)
                for chunk in _chunks(pending, batch_size):
                    for item in chunk:
                        sample_started = time.perf_counter()
                        if task == "chartqa":
                            record, score = _chartqa_prediction(model, item, generation)
                            chart_scores.append(score)
                        else:
                            record, em, f1 = _drop_prediction(model, item, generation)
                            drop_em.append(em)
                            drop_f1.append(f1)
                        handle.write(
                            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        handle.flush()
                        completed += 1
                        progress_record = {
                            "sample_index": completed - 1,
                            "sample_id": str(item["id"]),
                            "completed_samples": completed,
                            "total_samples": len(items),
                            "elapsed_seconds": time.perf_counter() - sample_started,
                            "output_characters": len(str(record["raw_prediction"])),
                            "output_tokens": _prediction_token_count(
                                model, str(record["raw_prediction"])
                            ),
                            "resources": _resource_snapshot(model),
                        }
                        progress_handle.write(
                            json.dumps(progress_record, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                        progress_handle.flush()
                        if completed % checkpoint_interval == 0:
                            os.fsync(handle.fileno())
                            os.fsync(progress_handle.fileno())
                            gc.collect()
            handle.flush()
            os.fsync(handle.fileno())
            progress_handle.flush()
            os.fsync(progress_handle.fileno())
        os.replace(incomplete, output)
    except BaseException:
        if not resume_partial:
            if incomplete.exists():
                incomplete.unlink()
            if progress.exists():
                progress.unlink()
        raise

    scores = (
        {"relaxed_accuracy": sum(chart_scores) / len(chart_scores)}
        if task == "chartqa"
        else {
            "em": sum(drop_em) / len(drop_em),
            "f1": sum(drop_f1) / len(drop_f1),
        }
    )
    generation_record = dict(generation)
    generation_record["batch_size"] = batch_size
    generation_record["message_order"] = (
        "question_then_image" if task == "chartqa" else "passage_then_question"
    )
    return {
        "task": task,
        "split": split,
        "num_samples": len(items),
        "generation": generation_record,
        "metrics": scores,
        "predictions_path": str(output),
        "progress_path": str(progress),
        "resumed_samples": len(existing_records),
        "complete": True,
    }
