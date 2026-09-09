"""ChartQA human-split loading and Qwen soft-prefix data adaptation."""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from skillopt.softprefix.data import (
    EncodedExample,
    apply_docvqa_image_budget,
    qwen_image_patch_size,
    resolve_docvqa_image_token_budget,
)


CHARTQA_SOURCE = "https://github.com/vis-nlp/ChartQA"
VALID_SPLITS = {"train", "val", "test"}
_LOGICAL_SPLITS = {"train": "train", "val": "validation", "test": "test"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(items: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(item["id"]) for item in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item["image_name"]).strip().casefold(),
        " ".join(str(item["question"]).split()).casefold(),
        " ".join(str(item["answer"]).split()).casefold(),
    )


def resolve_chartqa_dataset_root(path: str | os.PathLike[str]) -> Path:
    """Resolve either the cloned repository root or ``ChartQA Dataset``."""
    root = Path(path).expanduser().resolve()
    nested = root / "ChartQA Dataset"
    if nested.is_dir():
        root = nested
    if not all((root / split).is_dir() for split in VALID_SPLITS):
        raise FileNotFoundError(
            f"ChartQA root must contain train/val/test directories: {root}"
        )
    return root


def load_chartqa_human_split(
    dataset_root: str | os.PathLike[str],
    split: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load only the human-authored annotations from one official split."""
    split = str(split).strip().lower()
    if split not in VALID_SPLITS:
        raise ValueError(f"ChartQA split must be one of {sorted(VALID_SPLITS)}")
    if limit is not None and int(limit) < 1:
        raise ValueError("limit must be positive when provided")
    root = resolve_chartqa_dataset_root(dataset_root)
    annotation_path = root / split / f"{split}_human.json"
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    if limit is not None:
        rows = rows[: int(limit)]
    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        image_name = str(row["imgname"])
        image_path = root / split / "png" / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"ChartQA image is missing: {image_path}")
        items.append(
            {
                "id": f"chartqa:{split}:{index:06d}",
                "split": _LOGICAL_SPLITS[split],
                "source_split": split,
                "question": str(row["query"]),
                "answer": str(row["label"]),
                "image_name": image_name,
                "image_path": str(image_path),
                "source_type": "human",
            }
        )
    return items


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


def select_chartqa_transfer_splits(
    dataset_root: str | os.PathLike[str],
    *,
    train_size: int | None = None,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Use official human splits and seed-controlled sampling only for train."""
    root = resolve_chartqa_dataset_root(dataset_root)
    all_splits = {
        "train": load_chartqa_human_split(root, "train"),
        "validation": load_chartqa_human_split(root, "val"),
        "test": load_chartqa_human_split(root, "test"),
    }
    selected = {
        "train": _deterministic_sample(
            all_splits["train"], size=train_size, seed=int(seed)
        ),
        "validation": all_splits["validation"],
        "test": all_splits["test"],
    }
    overlap_counts = {
        f"{left}_{right}": len(
            {_fingerprint(item) for item in all_splits[left]}
            & {_fingerprint(item) for item in all_splits[right]}
        )
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    }
    if any(overlap_counts.values()):
        raise RuntimeError(f"ChartQA official splits contain direct overlap: {overlap_counts}")

    source_files = {
        logical: {
            "source_split": source_split,
            "path": str(root / source_split / f"{source_split}_human.json"),
            "sha256": _sha256_file(
                root / source_split / f"{source_split}_human.json"
            ),
            "source_samples": len(all_splits[logical]),
            "selected_samples": len(selected[logical]),
            "selected_ids_sha256": _ids_sha256(selected[logical]),
        }
        for logical, source_split in (
            ("train", "train"),
            ("validation", "val"),
            ("test", "test"),
        )
    }
    manifest = {
        "schema_version": 1,
        "task": "chartqa",
        "source": CHARTQA_SOURCE,
        "local_path": str(root),
        "splits": source_files,
        "filtering": {
            "human_authored_only": True,
            "annotation_pattern": "{split}_human.json",
            "csv_or_bbox_inputs": False,
        },
        "selection": {
            "seed": int(seed),
            "policy": (
                "seeded Python random shuffle of official human train, then first K; "
                "official human validation and test remain complete"
            ),
            "requested_train_size": train_size,
        },
        "direct_overlap_counts": overlap_counts,
        "prompt": {
            "modalities": ["question", "image"],
            "message_order": "question_then_image",
            "enable_thinking": False,
        },
        "metrics": ["chartqa_relaxed_accuracy_5pct"],
    }
    return selected, manifest


def build_chartqa_messages(
    item: dict[str, Any],
    *,
    image_detail: str = "auto",
) -> list[dict[str, Any]]:
    """Build Qwen messages with question before image and no auxiliary table data."""
    image_path = str(item.get("image_path", "")).strip()
    if not image_path:
        raise ValueError(f"ChartQA item {item.get('id', '<unknown>')} has no image_path")
    image: dict[str, Any] = {
        "type": "image",
        "image": f"file://{os.path.abspath(image_path)}",
    }
    if image_detail and image_detail != "auto":
        image["detail"] = image_detail
    question = str(item["question"]).strip()
    return [
        {
            "role": "system",
            "content": (
                "Answer the chart question from the image only. Return the shortest "
                "supported answer inside <answer>...</answer>."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                    + "\n\nReturn the final answer inside <answer>...</answer>.",
                },
                image,
            ],
        },
    ]


def build_chartqa_target(tokenizer: Any, item: dict[str, Any]) -> str:
    target = f"<answer>{str(item['answer']).strip()}</answer>"
    eos = getattr(tokenizer, "eos_token", None)
    return target + (eos or "")


class ChartQAPrefixDataset:
    """Qwen processor adapter yielding the existing ``EncodedExample`` format."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        processor: Any,
        tokenizer: Any,
        *,
        max_prompt_tokens: int,
        max_target_tokens: int,
        image_detail: str = "auto",
        max_image_tokens: int = 0,
    ) -> None:
        self.items = list(items)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.image_detail = str(image_detail)
        self.max_image_tokens = resolve_docvqa_image_token_budget(
            max_prompt_tokens=self.max_prompt_tokens,
            configured_max_image_tokens=int(max_image_tokens or 0),
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> EncodedExample:
        item = self.items[index]
        messages = apply_docvqa_image_budget(
            build_chartqa_messages(item, image_detail=self.image_detail),
            max_image_tokens=self.max_image_tokens,
            image_patch_size=qwen_image_patch_size(self.processor),
        )
        prompt_inputs = self._encode_prompt(messages)
        target_ids = self.tokenizer(
            build_chartqa_target(self.tokenizer, item),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens,
        )["input_ids"]
        prompt_ids = prompt_inputs["input_ids"]
        if len(prompt_ids) > self.max_prompt_tokens:
            raise ValueError(
                f"ChartQA item {item.get('id', index)!r} encoded to {len(prompt_ids)} "
                f"prompt tokens, exceeding {self.max_prompt_tokens}"
            )
        input_ids = prompt_ids + target_ids
        attention_mask = prompt_inputs["attention_mask"] + [1] * len(target_ids)
        mm_token_type_ids = prompt_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None and target_ids:
            import torch

            target_token_types = torch.zeros(
                (mm_token_type_ids.shape[0], len(target_ids)),
                dtype=mm_token_type_ids.dtype,
                device=mm_token_type_ids.device,
            )
            prompt_inputs["mm_token_type_ids"] = torch.cat(
                [mm_token_type_ids, target_token_types], dim=1
            )
        extra_inputs = {
            key: value
            for key, value in prompt_inputs.items()
            if key not in {"input_ids", "attention_mask"}
        }
        return EncodedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=[-100] * len(prompt_ids) + target_ids,
            extra_inputs=extra_inputs,
        )

    def _encode_prompt(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "ChartQA soft-prefix adaptation requires qwen-vl-utils"
            ) from exc
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=qwen_image_patch_size(self.processor),
        )
        kwargs: dict[str, Any] = {
            "text": [text],
            "images": image_inputs,
            "do_resize": False,
            "padding": False,
            "return_tensors": "pt",
        }
        if video_inputs:
            kwargs["videos"] = video_inputs
        encoded = self.processor(**kwargs)
        return {
            key: value[0].tolist() if key in {"input_ids", "attention_mask"} else value
            for key, value in encoded.items()
            if value is not None
        }


def chartqa_answer_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    """Small audit helper; categories are descriptive, not evaluation classes."""
    categories = Counter(
        "numeric" if _is_float(item["answer"]) else "string" for item in items
    )
    return dict(sorted(categories.items()))


def _is_float(value: Any) -> bool:
    try:
        float(str(value).replace(",", ""))
        return True
    except ValueError:
        return False
