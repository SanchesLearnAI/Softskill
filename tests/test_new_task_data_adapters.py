"""Stage-1 tests for ChartQA and DROP adapters and official-style metrics."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from skillopt.envs.chartqa.data import (
    ChartQAPrefixDataset,
    _deterministic_sample as chartqa_deterministic_sample,
    build_chartqa_messages,
    build_chartqa_target,
    load_chartqa_human_split,
    select_chartqa_transfer_splits,
)
from skillopt.envs.chartqa.evaluator import relaxed_accuracy
from skillopt.envs.drop.data import (
    answer_json_to_strings,
    build_drop_messages,
    build_drop_prompt,
    build_drop_target,
    load_drop_split,
    select_drop_transfer_splits,
)
from skillopt.envs.drop.evaluator import evaluate as evaluate_drop
from skillopt.envs.drop.evaluator import get_metrics


CHARTQA_ROOT = os.environ.get("SOFTSKILL_CHARTQA_ROOT", "")
DROP_ROOT = os.environ.get("SOFTSKILL_DROP_ROOT", "")


class _Tokenizer:
    chat_template = "qwen-test-template"
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.template_kwargs: dict = {}

    def apply_chat_template(self, messages, **kwargs):
        del messages
        self.template_kwargs = kwargs
        return "rendered prompt"

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": list(range(max(1, min(len(str(text)), 12))))}


class _Vector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class _Matrix:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return _Vector(self.values)


class _ChartProcessor:
    image_processor = types.SimpleNamespace(patch_size=14)

    def __init__(self) -> None:
        self.template_kwargs: dict = {}

    def apply_chat_template(self, messages, **kwargs):
        del messages
        self.template_kwargs = kwargs
        return "rendered multimodal prompt"

    def __call__(self, **kwargs):
        if kwargs["do_resize"] is not False:
            raise AssertionError("ChartQA processor must not resize twice")
        return {
            "input_ids": _Matrix([1, 2, 3]),
            "attention_mask": _Matrix([1, 1, 1]),
        }


class NewTaskAdapterUnitTests(unittest.TestCase):
    def test_chartqa_relaxed_accuracy_numeric_tolerance_and_text(self) -> None:
        self.assertEqual(relaxed_accuracy("104.9", "100"), 1.0)
        self.assertEqual(relaxed_accuracy("105.1", "100"), 0.0)
        self.assertEqual(relaxed_accuracy("  Blue  ", "blue"), 1.0)
        self.assertEqual(relaxed_accuracy("green", "blue"), 0.0)

    def test_chartqa_prompt_is_question_then_image_and_disables_thinking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "chart.png"
            image_path.write_bytes(b"adapter-test")
            item = {
                "id": "chartqa:test:0",
                "question": "Which bar is highest?",
                "answer": "Blue",
                "image_path": str(image_path),
            }
            messages = build_chartqa_messages(item)
            self.assertEqual(
                [part["type"] for part in messages[1]["content"]],
                ["text", "image"],
            )
            self.assertTrue(messages[1]["content"][0]["text"].startswith(item["question"]))
            self.assertTrue(all("csv" not in str(message).lower() for message in messages))

            fake_qwen_utils = types.SimpleNamespace(
                process_vision_info=lambda *args, **kwargs: (["image"], [])
            )
            with patch.dict(sys.modules, {"qwen_vl_utils": fake_qwen_utils}):
                processor = _ChartProcessor()
                dataset = object.__new__(ChartQAPrefixDataset)
                dataset.processor = processor
                dataset._encode_prompt(messages)
            self.assertIs(processor.template_kwargs["enable_thinking"], False)

    def test_chartqa_seeded_sampling_is_reproducible(self) -> None:
        items = [{"id": str(index)} for index in range(20)]
        first = chartqa_deterministic_sample(items, size=7, seed=13)
        second = chartqa_deterministic_sample(items, size=7, seed=13)
        different = chartqa_deterministic_sample(items, size=7, seed=14)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_drop_answer_forms_and_multispan_target_remain_structured(self) -> None:
        self.assertEqual(
            answer_json_to_strings({"number": "12", "date": {}, "spans": []}),
            (["12"], "number"),
        )
        self.assertEqual(
            answer_json_to_strings(
                {
                    "number": "",
                    "date": {"day": "4", "month": "July", "year": "1776"},
                    "spans": [],
                }
            ),
            (["4 July 1776"], "date"),
        )
        self.assertEqual(
            answer_json_to_strings(
                {"number": "", "date": {}, "spans": ["Boston"]}
            ),
            (["Boston"], "single-span"),
        )
        values, answer_type = answer_json_to_strings(
            {"number": "", "date": {}, "spans": ["Boston", "Denver"]}
        )
        self.assertEqual((values, answer_type), (["Boston", "Denver"], "multi-span"))

        target = build_drop_target(
            _Tokenizer(), {"answer_values": values, "answer_type": answer_type}
        )
        payload = target.removeprefix("<answer>").removesuffix("</answer><eos>")
        self.assertEqual(json.loads(payload), ["Boston", "Denver"])

    def test_drop_prompt_disables_thinking(self) -> None:
        tokenizer = _Tokenizer()
        prompt = build_drop_prompt(
            tokenizer,
            {"passage": "Alice scored 12 points.", "question": "How many points?"},
        )
        self.assertEqual(prompt, "rendered prompt")
        self.assertIs(tokenizer.template_kwargs["enable_thinking"], False)

    def test_drop_official_metrics_single_number_and_multispan(self) -> None:
        self.assertEqual(get_metrics("the touchdown", "touchdown"), (1.0, 1.0))
        self.assertEqual(get_metrics("12.0", "12"), (1.0, 1.0))
        self.assertEqual(
            get_metrics(["Denver", "Boston"], ["Boston", "Denver"]),
            (1.0, 1.0),
        )
        result = evaluate_drop(
            '<answer>["Boston", "Denver"]</answer>',
            [
                {
                    "number": "",
                    "date": {"day": "", "month": "", "year": ""},
                    "spans": ["Denver", "Boston"],
                }
            ],
        )
        self.assertEqual(result["em"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["predicted_answer"], ["Boston", "Denver"])


@unittest.skipUnless(CHARTQA_ROOT, "SOFTSKILL_CHARTQA_ROOT is not set")
class ChartQARealDataTests(unittest.TestCase):
    def test_human_splits_prompt_and_no_direct_overlap(self) -> None:
        samples = load_chartqa_human_split(CHARTQA_ROOT, "train", limit=3)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(item["source_type"] == "human" for item in samples))
        self.assertTrue(all(Path(item["image_path"]).is_file() for item in samples))
        self.assertTrue(all(item["answer"] for item in samples))
        self.assertEqual(build_chartqa_messages(samples[0])[1]["content"][0]["type"], "text")
        self.assertIn(str(samples[0]["answer"]), build_chartqa_target(_Tokenizer(), samples[0]))

        selected, audit = select_chartqa_transfer_splits(
            CHARTQA_ROOT, train_size=8, seed=1
        )
        self.assertEqual(len(selected["train"]), 8)
        self.assertEqual(audit["splits"]["train"]["source_samples"], 7398)
        self.assertEqual(audit["splits"]["validation"]["source_samples"], 960)
        self.assertEqual(audit["splits"]["test"]["source_samples"], 1250)
        self.assertEqual(set(audit["direct_overlap_counts"].values()), {0})


@unittest.skipUnless(DROP_ROOT, "SOFTSKILL_DROP_ROOT is not set")
class DropRealDataTests(unittest.TestCase):
    def test_splits_labels_prompt_and_no_direct_overlap(self) -> None:
        samples = load_drop_split(DROP_ROOT, "train", limit=3)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(item["answer_values"] for item in samples))
        messages = build_drop_messages(samples[0])
        self.assertIn("Passage:", messages[1]["content"])
        self.assertIn("Question:", messages[1]["content"])

        selected, audit = select_drop_transfer_splits(
            DROP_ROOT, train_size=32, seed=1
        )
        self.assertEqual(len(selected["train"]), 32)
        self.assertEqual(audit["splits"]["train"]["source_samples"], 77409)
        self.assertEqual(audit["splits"]["train"]["usable_answer_samples"], 77400)
        self.assertEqual(audit["splits"]["train"]["post_filter_samples"], 77399)
        self.assertEqual(audit["splits"]["validation"]["source_samples"], 9536)
        self.assertEqual(
            audit["filtering"]["excluded_train_rows_without_any_nonempty_gold_answer"],
            9,
        )
        self.assertEqual(
            audit["filtering"][
                "excluded_train_rows_matching_validation_passage_and_question"
            ],
            1,
        )
        self.assertEqual(set(audit["direct_overlap_counts"].values()), {0})


if __name__ == "__main__":
    unittest.main()
