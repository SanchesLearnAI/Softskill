"""Directed mock tests for stage-3C1 independent prediction evaluation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from skillopt.behavior_compression.new_task_transfer.v1.evaluation import (
    evaluate_transfer_predictions,
)


class _Tokenizer:
    chat_template = "mock-qwen-template"

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, dict(kwargs)))
        return "rendered-drop-prompt"


class _EvalFlag:
    def __init__(self) -> None:
        self.was_called = False

    def eval(self) -> None:
        self.was_called = True


class _MockGenerator:
    def __init__(self, responses: list[str | BaseException]) -> None:
        self.responses = list(responses)
        self.tokenizer = _Tokenizer()
        self.model = _EvalFlag()
        self.message_calls: list[tuple[list[dict], dict]] = []
        self.prompt_calls: list[tuple[str, dict]] = []

    def _next(self) -> str:
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def generate_from_messages(self, messages, **kwargs) -> str:
        self.message_calls.append((messages, dict(kwargs)))
        return self._next()

    def generate_from_prompt(self, prompt, **kwargs) -> str:
        self.prompt_calls.append((str(prompt), dict(kwargs)))
        return self._next()


def _chart_item(index: int, answer: str, *, question: str = "Question?") -> dict:
    return {
        "id": f"chartqa:test:{index}",
        "split": "test",
        "source_type": "human",
        "question": question,
        "answer": answer,
        "image_path": f"/private/chart-{index}.png",
    }


def _answer(
    *,
    number: str = "",
    spans: list[str] | None = None,
    day: str = "",
    month: str = "",
    year: str = "",
) -> dict:
    return {
        "number": number,
        "date": {"day": day, "month": month, "year": year},
        "spans": list(spans or []),
    }


def _drop_item(
    index: int,
    candidate: dict,
    answer_type: str,
    *,
    passage: str = "A private passage.",
) -> dict:
    return {
        "id": f"drop:{index}",
        "split": "validation",
        "passage": passage,
        "question": "What is the answer?",
        "candidate_answers": [candidate],
        "answer_type": answer_type,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


class IndependentEvaluationTests(unittest.TestCase):
    def test_chartqa_records_predictions_and_relaxed_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            metrics = evaluate_transfer_predictions(
                model=_MockGenerator(["<answer>104.9</answer>", "<answer>red</answer>"]),
                task="chartqa",
                dataset=[_chart_item(0, "100"), _chart_item(1, "blue")],
                split="test",
                batch_size=2,
                predictions_path=output,
            )
            rows = _read_jsonl(output)
            self.assertEqual(metrics["metrics"]["relaxed_accuracy"], 0.5)
            self.assertEqual(rows[0]["normalized_prediction"], "104.9")
            self.assertEqual(rows[0]["relaxed_accuracy"], 1.0)
            self.assertEqual(rows[1]["relaxed_accuracy"], 0.0)

    def test_chartqa_question_precedes_image_and_thinking_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generator = _MockGenerator(["<answer>blue</answer>"])
            metrics = evaluate_transfer_predictions(
                model=generator,
                task="chartqa",
                dataset=[_chart_item(0, "blue", question="Which bar?")],
                split="test",
                predictions_path=Path(temporary) / "predictions.jsonl",
            )
            messages, kwargs = generator.message_calls[0]
            content = messages[1]["content"]
            self.assertEqual([part["type"] for part in content], ["text", "image"])
            self.assertTrue(content[0]["text"].startswith("Which bar?"))
            self.assertEqual(kwargs["max_new_tokens"], 64)
            self.assertEqual(kwargs["stop_strings"], ["</answer>"])
            self.assertIs(kwargs["use_cache"], True)
            self.assertIs(metrics["generation"]["enable_thinking"], False)
            self.assertEqual(metrics["generation"]["stop_strings"], ["</answer>"])
            self.assertIs(metrics["generation"]["use_cache"], True)
            self.assertEqual(metrics["generation"]["message_order"], "question_then_image")

    def test_drop_single_answer_em_f1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            metrics = evaluate_transfer_predictions(
                model=_MockGenerator(["<answer>the touchdown</answer>"]),
                task="drop",
                dataset=[_drop_item(0, _answer(spans=["touchdown"]), "single-span")],
                split="validation",
                predictions_path=output,
            )
            self.assertEqual(metrics["metrics"], {"em": 1.0, "f1": 1.0})
            self.assertEqual(_read_jsonl(output)[0]["answer_type"], "single-span")

    def test_drop_numeric_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            metrics = evaluate_transfer_predictions(
                model=_MockGenerator(["<answer>12.0</answer>"]),
                task="drop",
                dataset=[_drop_item(0, _answer(number="12"), "number")],
                split="validation",
                predictions_path=output,
            )
            self.assertEqual(metrics["metrics"], {"em": 1.0, "f1": 1.0})
            self.assertEqual(_read_jsonl(output)[0]["parsed_prediction"], "12.0")

    def test_drop_multispan_json_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            evaluate_transfer_predictions(
                model=_MockGenerator(['<answer>["Boston", "Denver"]</answer>']),
                task="drop",
                dataset=[
                    _drop_item(
                        0,
                        _answer(spans=["Denver", "Boston"]),
                        "multi-span",
                    )
                ],
                split="validation",
                predictions_path=output,
            )
            row = _read_jsonl(output)[0]
            self.assertEqual(row["parsed_prediction"], ["Boston", "Denver"])
            self.assertEqual((row["em"], row["f1"]), (1.0, 1.0))

    def test_drop_invalid_json_uses_safe_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            metrics = evaluate_transfer_predictions(
                model=_MockGenerator(['<answer>["Boston",</answer>']),
                task="drop",
                dataset=[_drop_item(0, _answer(spans=["Boston"]), "single-span")],
                split="validation",
                predictions_path=output,
            )
            row = _read_jsonl(output)[0]
            self.assertIsInstance(row["parsed_prediction"], str)
            self.assertEqual(metrics["num_samples"], 1)

    def test_predictions_retain_original_sample_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            items = [_chart_item(index, str(index)) for index in (4, 1, 9)]
            evaluate_transfer_predictions(
                model=_MockGenerator([f"<answer>{index}</answer>" for index in (4, 1, 9)]),
                task="chartqa",
                dataset=items,
                split="test",
                batch_size=2,
                predictions_path=output,
            )
            self.assertEqual(
                [row["sample_id"] for row in _read_jsonl(output)],
                [item["id"] for item in items],
            )

    def test_outputs_exclude_image_passage_and_prompt_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chart_output = root / "chart.jsonl"
            chart_item = _chart_item(0, "blue", question="PRIVATE_CHART_QUESTION")
            evaluate_transfer_predictions(
                model=_MockGenerator(["<answer>blue</answer>"]),
                task="chartqa",
                dataset=[chart_item],
                split="test",
                predictions_path=chart_output,
            )
            drop_output = root / "drop.jsonl"
            evaluate_transfer_predictions(
                model=_MockGenerator(["<answer>4 July 1776</answer>"]),
                task="drop",
                dataset=[
                    _drop_item(
                        0,
                        _answer(day="4", month="July", year="1776"),
                        "date",
                        passage="PRIVATE_COMPLETE_PASSAGE",
                    )
                ],
                split="validation",
                predictions_path=drop_output,
            )
            serialized = chart_output.read_text("utf-8") + drop_output.read_text("utf-8")
            self.assertNotIn("PRIVATE_CHART_QUESTION", serialized)
            self.assertNotIn(chart_item["image_path"], serialized)
            self.assertNotIn("PRIVATE_COMPLETE_PASSAGE", serialized)
            self.assertNotIn("rendered-drop-prompt", serialized)

    def test_generation_failure_never_publishes_complete_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            with self.assertRaisesRegex(RuntimeError, "mock generation failure"):
                evaluate_transfer_predictions(
                    model=_MockGenerator(
                        ["<answer>first</answer>", RuntimeError("mock generation failure")]
                    ),
                    task="chartqa",
                    dataset=[_chart_item(0, "first"), _chart_item(1, "second")],
                    split="test",
                    predictions_path=output,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(".*.incomplete.*")), [])

    def test_resumable_failure_preserves_progress_and_continues_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            items = [_chart_item(0, "first"), _chart_item(1, "second")]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                evaluate_transfer_predictions(
                    model=_MockGenerator(
                        ["<answer>first</answer>", RuntimeError("interrupted")]
                    ),
                    task="chartqa",
                    dataset=items,
                    split="test",
                    predictions_path=output,
                    resume_partial=True,
                    checkpoint_interval=1,
                )
            partial = output.with_name(f".{output.name}.partial")
            progress = output.with_name(f".{output.name}.progress.jsonl")
            self.assertFalse(output.exists())
            self.assertEqual([row["sample_id"] for row in _read_jsonl(partial)], [items[0]["id"]])
            self.assertEqual(len(_read_jsonl(progress)), 1)

            result = evaluate_transfer_predictions(
                model=_MockGenerator(["<answer>second</answer>"]),
                task="chartqa",
                dataset=items,
                split="test",
                predictions_path=output,
                resume_partial=True,
                checkpoint_interval=1,
            )
            self.assertEqual(result["resumed_samples"], 1)
            self.assertFalse(partial.exists())
            self.assertEqual(
                [row["sample_id"] for row in _read_jsonl(output)],
                [item["id"] for item in items],
            )


if __name__ == "__main__":
    unittest.main()
