"""DROP loading, split isolation, and Qwen soft-prefix data adaptation."""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from skillopt.softprefix.data import EncodedExample, _apply_text_chat_template


DROP_SOURCE = "https://allennlp.org/drop"
_SOURCE_FILES = {"train": "train", "validation": "dev"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(items: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(item["id"]) for item in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _direct_fingerprint(item: dict[str, Any]) -> tuple[str, str]:
    return (
        " ".join(str(item["passage"]).split()).casefold(),
        " ".join(str(item["question"]).split()).casefold(),
    )


def resolve_drop_dataset_root(path: str | os.PathLike[str]) -> Path:
    """Resolve the extraction parent, ``raw`` folder, or dataset folder."""
    root = Path(path).expanduser().resolve()
    candidates = [
        root,
        root / "drop_dataset",
        root / "raw" / "drop_dataset",
    ]
    for candidate in candidates:
        required = [
            candidate / "drop_dataset_train.json",
            candidate / "drop_dataset_dev.json",
        ]
        if all(file.is_file() for file in required):
            return candidate
    raise FileNotFoundError(
        "DROP root must resolve to drop_dataset_train.json and "
        f"drop_dataset_dev.json: {root}"
    )


def answer_json_to_strings(answer: dict[str, Any]) -> tuple[list[str], str]:
    """Convert the released answer JSON without flattening multi-span answers."""
    if answer.get("number") not in (None, ""):
        return [str(answer["number"])], "number"
    spans = answer.get("spans") or []
    if spans:
        values = [str(value) for value in spans]
        return values, "single-span" if len(values) == 1 else "multi-span"
    date = answer.get("date") or {}
    if date and any(date.get(part) for part in ("day", "month", "year")):
        value = f"{date.get('day', '')} {date.get('month', '')} {date.get('year', '')}"
        return [" ".join(value.split())], "date"
    raise ValueError(f"DROP answer has no number, spans, or date: {answer!r}")


def _answer_is_nonempty(answer: dict[str, Any]) -> bool:
    try:
        answer_json_to_strings(answer)
    except ValueError:
        return False
    return True


def _source_answer_counts(path: Path) -> tuple[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = [qa for record in raw.values() for qa in record.get("qa_pairs", [])]
    usable = sum(
        any(
            _answer_is_nonempty(answer)
            for answer in [qa.get("answer") or {}, *(qa.get("validated_answers") or [])]
        )
        for qa in rows
    )
    return len(rows), usable


def _flatten_drop_file(path: Path, *, split: str) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for passage_id, passage_record in raw.items():
        passage = str(passage_record["passage"])
        for qa in passage_record.get("qa_pairs", []):
            candidates = [
                answer
                for answer in [qa.get("answer") or {}, *(qa.get("validated_answers") or [])]
                if _answer_is_nonempty(answer)
            ]
            if not candidates:
                continue
            canonical_answer = candidates[0]
            answer_values, answer_type = answer_json_to_strings(canonical_answer)
            items.append(
                {
                    "id": str(qa["query_id"]),
                    "split": split,
                    "source_split": _SOURCE_FILES[split],
                    "passage_id": str(passage_id),
                    "passage": passage,
                    "question": str(qa["question"]),
                    "answer": canonical_answer,
                    "answer_values": answer_values,
                    "answer_type": answer_type,
                    "candidate_answers": candidates,
                }
            )
    return items


def load_drop_split(
    dataset_root: str | os.PathLike[str],
    split: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    split = str(split).strip().lower()
    if split == "dev":
        split = "validation"
    if split not in _SOURCE_FILES:
        raise ValueError("DROP split must be train, validation, or dev")
    if limit is not None and int(limit) < 1:
        raise ValueError("limit must be positive when provided")
    root = resolve_drop_dataset_root(dataset_root)
    items = _flatten_drop_file(
        root / f"drop_dataset_{_SOURCE_FILES[split]}.json",
        split=split,
    )
    return items if limit is None else items[: int(limit)]


def _deterministic_sample(
    items: list[dict[str, Any]],
    *,
    size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if size is None:
        return list(items)
    if int(size) < 1 or int(size) > len(items):
        raise ValueError(f"sample size must be in [1, {len(items)}], got {size}")
    order = list(range(len(items)))
    random.Random(int(seed)).shuffle(order)
    return [items[index] for index in order[: int(size)]]


def select_drop_transfer_splits(
    dataset_root: str | os.PathLike[str],
    *,
    train_size: int | None = None,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Sample official train and retain official dev as untouched validation.

    The release contains one duplicated passage/question pair across train and
    dev.  The train copy is deterministically excluded so direct content cannot
    leak into validation; the official validation set remains unchanged.
    """
    root = resolve_drop_dataset_root(dataset_root)
    official_train = load_drop_split(root, "train")
    validation = load_drop_split(root, "validation")
    train_source_samples, train_usable_samples = _source_answer_counts(
        root / "drop_dataset_train.json"
    )
    validation_source_samples, validation_usable_samples = _source_answer_counts(
        root / "drop_dataset_dev.json"
    )
    validation_fingerprints = {_direct_fingerprint(item) for item in validation}
    train_pool = [
        item
        for item in official_train
        if _direct_fingerprint(item) not in validation_fingerprints
    ]
    excluded = len(official_train) - len(train_pool)
    train = _deterministic_sample(train_pool, size=train_size, seed=int(seed))
    overlap_count = len(
        {_direct_fingerprint(item) for item in train}
        & validation_fingerprints
    )
    id_overlap_count = len(
        {str(item["id"]) for item in train}
        & {str(item["id"]) for item in validation}
    )
    if overlap_count or id_overlap_count:
        raise RuntimeError(
            "DROP train/validation overlap remains after filtering: "
            f"content={overlap_count}, ids={id_overlap_count}"
        )

    selected = {"train": train, "validation": validation}
    manifest = {
        "schema_version": 1,
        "task": "drop",
        "source": DROP_SOURCE,
        "local_path": str(root),
        "splits": {
            "train": {
                "source_split": "train",
                "path": str(root / "drop_dataset_train.json"),
                "sha256": _sha256_file(root / "drop_dataset_train.json"),
                "source_samples": train_source_samples,
                "usable_answer_samples": train_usable_samples,
                "post_filter_samples": len(train_pool),
                "selected_samples": len(train),
                "selected_ids_sha256": _ids_sha256(train),
            },
            "validation": {
                "source_split": "dev",
                "path": str(root / "drop_dataset_dev.json"),
                "sha256": _sha256_file(root / "drop_dataset_dev.json"),
                "source_samples": validation_source_samples,
                "usable_answer_samples": validation_usable_samples,
                "post_filter_samples": len(validation),
                "selected_samples": len(validation),
                "selected_ids_sha256": _ids_sha256(validation),
            },
        },
        "filtering": {
            "answer_forms_retained": [
                "number",
                "date",
                "single-span",
                "multi-span",
            ],
            "excluded_train_rows_without_any_nonempty_gold_answer": (
                train_source_samples - train_usable_samples
            ),
            "excluded_train_rows_matching_validation_passage_and_question": excluded,
            "validation_unchanged": True,
        },
        "selection": {
            "seed": int(seed),
            "policy": (
                "exclude direct validation duplicates, seeded Python random shuffle "
                "of remaining official train, then first K; official dev stays complete"
            ),
            "requested_train_size": train_size,
        },
        "direct_overlap_counts": {
            "train_validation_content": overlap_count,
            "train_validation_query_id": id_overlap_count,
        },
        "answer_type_counts": {
            split: dict(sorted(Counter(item["answer_type"] for item in rows).items()))
            for split, rows in selected.items()
        },
        "prompt": {
            "modalities": ["passage", "question"],
            "message_order": "passage_then_question",
            "enable_thinking": False,
            "multi_span_output": "JSON string list inside <answer> tags",
        },
        "metrics": ["drop_exact_match", "drop_f1"],
    }
    return selected, manifest


def build_drop_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    passage = str(item["passage"]).strip()
    question = str(item["question"]).strip()
    return [
        {
            "role": "system",
            "content": (
                "Answer only from the passage. For a multi-span answer, return a "
                "valid JSON list of strings. Put the final answer inside "
                "<answer>...</answer>."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Passage:\n{passage}\n\nQuestion:\n{question}\n\n"
                "Return the final answer inside <answer>...</answer>."
            ),
        },
    ]


def build_drop_prompt(tokenizer: Any, item: dict[str, Any]) -> str:
    """Render through the shared Qwen chat helper with thinking forced off."""
    return _apply_text_chat_template(
        tokenizer,
        build_drop_messages(item),
        enable_thinking=False,
        add_generation_prompt=True,
    )


def build_drop_target(tokenizer: Any, item: dict[str, Any]) -> str:
    values = [str(value) for value in item["answer_values"]]
    answer_text = (
        json.dumps(values, ensure_ascii=False) if len(values) > 1 else values[0]
    )
    target = f"<answer>{answer_text}</answer>"
    eos = getattr(tokenizer, "eos_token", None)
    return target + (eos or "")


class DropPrefixDataset:
    """Text-only DROP adapter yielding the v2 ``EncodedExample`` format."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        tokenizer: Any,
        *,
        max_prompt_tokens: int,
        max_target_tokens: int,
    ) -> None:
        self.items = list(items)
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_target_tokens = int(max_target_tokens)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> EncodedExample:
        item = self.items[index]
        prompt_ids = self.tokenizer(
            build_drop_prompt(self.tokenizer, item),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_tokens,
        )["input_ids"]
        target_ids = self.tokenizer(
            build_drop_target(self.tokenizer, item),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        return EncodedExample(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=[-100] * len(prompt_ids) + target_ids,
        )
