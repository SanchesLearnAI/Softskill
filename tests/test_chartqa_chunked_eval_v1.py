"""Directed tests for process-isolated ChartQA evaluation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skillopt.behavior_compression.new_task_transfer.v1.chartqa_chunked_eval import (
    chunk_paths,
    merge_chartqa_chunks,
    run_chartqa_chunk,
)
from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import file_sha256


def _items(split: str, answers: list[str]) -> list[dict]:
    source = "val" if split == "validation" else "test"
    return [
        {
            "id": f"chartqa:{source}:{index:06d}",
            "split": split,
            "source_type": "human",
            "question": f"Question {index}?",
            "answer": answer,
            "image_path": f"/mock/{source}-{index}.png",
        }
        for index, answer in enumerate(answers)
    ]


class _Tokenizer:
    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": str(text).split()}


class _ModelFlag:
    def eval(self) -> None:
        return None


class _Generator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.tokenizer = _Tokenizer()
        self.model = _ModelFlag()

    def generate_from_messages(self, messages, **kwargs):
        del messages, kwargs
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ChartQAChunkedEvaluationTests(unittest.TestCase):
    def test_validation_and_test_merge_without_ledger_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"chart prefix")
            digest = file_sha256(checkpoint)
            for split, answers in (("validation", ["10", "red"]), ("test", ["20", "blue", "3"])):
                dataset = _items(split, answers)
                model = _Generator([f"<answer>{answer}</answer>" for answer in answers])
                for start in range(0, len(dataset), 2):
                    run_chartqa_chunk(
                        model=model,
                        dataset=dataset,
                        output_dir=root,
                        checkpoint_path=checkpoint,
                        checkpoint_sha256=digest,
                        condition="i32",
                        seed=1,
                        split=split,
                        start=start,
                        chunk_size=2,
                    )
                merged = merge_chartqa_chunks(
                    dataset=dataset,
                    output_dir=root,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=digest,
                    condition="i32",
                    seed=1,
                    split=split,
                    chunk_size=2,
                )
                self.assertEqual(merged["metrics"], {"relaxed_accuracy": 1.0})
            ledger = _read_jsonl(root / "eval_metrics.jsonl")
            self.assertEqual([row["split"] for row in ledger], ["validation", "test"])
            self.assertTrue(ledger[1]["report_split"])

    def test_interrupted_chunk_resumes_in_a_new_model_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"chart prefix")
            dataset = _items("test", ["red", "blue"])
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_chartqa_chunk(
                    model=_Generator(["<answer>red</answer>", RuntimeError("interrupted")]),
                    dataset=dataset,
                    output_dir=root,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=file_sha256(checkpoint),
                    condition="f0",
                    seed=1,
                    split="test",
                    start=0,
                    chunk_size=2,
                )
            paths = chunk_paths(root, "test", 0, 2)
            self.assertFalse(paths["predictions"].exists())
            result = run_chartqa_chunk(
                model=_Generator(["<answer>blue</answer>"]),
                dataset=dataset,
                output_dir=root,
                checkpoint_path=checkpoint,
                checkpoint_sha256=file_sha256(checkpoint),
                condition="f0",
                seed=1,
                split="test",
                start=0,
                chunk_size=2,
            )
            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(len(_read_jsonl(paths["predictions"])), 2)

    def test_complete_chunk_reuses_manifest_without_generator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"chart prefix")
            dataset = _items("test", ["red"])
            kwargs = {
                "dataset": dataset,
                "output_dir": root,
                "checkpoint_path": checkpoint,
                "checkpoint_sha256": file_sha256(checkpoint),
                "condition": "f1",
                "seed": 1,
                "split": "test",
                "start": 0,
                "chunk_size": 1,
            }
            first = run_chartqa_chunk(model=_Generator(["<answer>red</answer>"]), **kwargs)
            reused = run_chartqa_chunk(model=None, **kwargs)
            self.assertFalse(first["reused"])
            self.assertTrue(reused["reused"])

    def test_missing_chunk_prevents_formal_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"chart prefix")
            with self.assertRaisesRegex(RuntimeError, "missing complete"):
                merge_chartqa_chunks(
                    dataset=_items("validation", ["red", "blue"]),
                    output_dir=root,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=file_sha256(checkpoint),
                    condition="f0",
                    seed=1,
                    split="validation",
                    chunk_size=2,
                )
            self.assertFalse((root / "predictions" / "validation.jsonl").exists())
            self.assertFalse((root / "eval_metrics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
