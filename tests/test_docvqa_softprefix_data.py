"""Tests for the canonical DocVQA soft-prefix prompt and image budgeting."""
from __future__ import annotations

import sys
import types
from pathlib import Path

from skillopt.softprefix.data import (
    DocVQAPrefixDataset,
    apply_docvqa_image_budget,
    build_docvqa_messages,
    resolve_docvqa_image_token_budget,
)


def test_docvqa_messages_match_rollout_question_then_image_order(tmp_path: Path) -> None:
    image_path = tmp_path / "document.png"
    image_path.write_bytes(b"not-decoded-by-this-test")

    messages = build_docvqa_messages(
        {"id": "doc-1", "question": "What is the invoice number?", "image_path": str(image_path)}
    )

    user_content = messages[1]["content"]
    assert [part["type"] for part in user_content] == ["text", "image"]
    assert user_content[0]["text"].startswith("What is the invoice number?")


def test_docvqa_training_chat_template_disables_thinking(monkeypatch) -> None:
    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        types.SimpleNamespace(process_vision_info=lambda *args, **kwargs: ([], [])),
    )

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "rendered"

        def __call__(self, **kwargs):
            del kwargs
            return {"input_ids": [[1]], "attention_mask": [[1]]}

    dataset = object.__new__(DocVQAPrefixDataset)
    dataset.processor = Processor()
    dataset._encode_prompt([{"role": "user", "content": []}])

    assert captured["kwargs"]["enable_thinking"] is False


def test_auto_docvqa_image_budget_leaves_room_for_text() -> None:
    assert resolve_docvqa_image_token_budget(max_prompt_tokens=16_384, configured_max_image_tokens=0) == 12_288


def test_configured_docvqa_image_budget_is_clamped_to_prompt_budget() -> None:
    assert resolve_docvqa_image_token_budget(max_prompt_tokens=8_192, configured_max_image_tokens=20_000) == 8_064


def test_apply_docvqa_image_budget_sets_qwen_pixel_cap() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "file:///tmp/page.png"},
                {"type": "text", "text": "question"},
            ],
        },
    ]

    updated = apply_docvqa_image_budget(messages, max_image_tokens=100, image_patch_size=14)

    image = updated[1]["content"][0]
    assert image["max_pixels"] == 100 * (14 * 2) ** 2
    assert image["image"] == "file:///tmp/page.png"
    assert "max_pixels" not in messages[1]["content"][0]
