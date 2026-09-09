"""Directed tests for process-isolated resumable DROP evaluation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import file_sha256
from skillopt.behavior_compression.new_task_transfer.v1.drop_chunked_eval import (
    chunk_paths,
    chunk_ranges,
    merge_drop_chunks,
    run_drop_chunk,
)


def _answer(value: str) -> dict:
    return {"number": value, "date": {"day": "", "month": "", "year": ""}, "spans": []}


def _items(count: int) -> list[dict]:
    return [
        {
            "id": f"drop:{index}",
            "split": "validation",
            "passage": f"Passage {index}",
            "question": "How many?",
            "candidate_answers": [_answer(str(index))],
            "answer_type": "number",
        }
        for index in range(count)
    ]


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return "rendered prompt"

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

    def generate_from_prompt(self, prompt, **kwargs):
        del prompt, kwargs
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class DropChunkedEvaluationTests(unittest.TestCase):
    def test_ranges_cover_dataset_exactly(self) -> None:
        self.assertEqual(chunk_ranges(5, 2), [(0, 2), (2, 4), (4, 5)])
        with self.assertRaises(ValueError):
            chunk_ranges(5, 0)

    def test_chunks_merge_in_order_and_publish_formal_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"prefix checkpoint")
            dataset = _items(5)
            model = _Generator([f"<answer>{index}</answer>" for index in range(5)])
            for start in (0, 2, 4):
                run_drop_chunk(
                    model=model,
                    dataset=dataset,
                    output_dir=root,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=file_sha256(checkpoint),
                    seed=1,
                    start=start,
                    chunk_size=2,
                )
            merged = merge_drop_chunks(
                dataset=dataset,
                output_dir=root,
                checkpoint_path=checkpoint,
                checkpoint_sha256=file_sha256(checkpoint),
                seed=1,
                chunk_size=2,
            )
            predictions = _read_jsonl(root / "predictions" / "validation.jsonl")
            self.assertEqual([row["sample_id"] for row in predictions], [item["id"] for item in dataset])
            self.assertEqual(merged["metrics"], {"em": 1.0, "f1": 1.0})
            self.assertEqual(merged["chunk_count"], 3)
            ledger = _read_jsonl(root / "eval_metrics.jsonl")
            self.assertEqual(ledger[0]["execution_mode"], "chunked_multi_process")
            self.assertEqual(ledger[0]["num_samples"], 5)

    def test_interrupted_chunk_resumes_without_duplicate_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"prefix checkpoint")
            dataset = _items(3)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_drop_chunk(
                    model=_Generator(["<answer>0</answer>", RuntimeError("interrupted")]),
                    dataset=dataset,
                    output_dir=root,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=file_sha256(checkpoint),
                    seed=1,
                    start=0,
                    chunk_size=3,
                )
            paths = chunk_paths(root, 0, 3)
            self.assertFalse(paths["predictions"].exists())
            self.assertEqual(len(_read_jsonl(paths["predictions"].with_name(".validation_000000_000003.jsonl.partial"))), 1)
            result = run_drop_chunk(
                model=_Generator(["<answer>1</answer>", "<answer>2</answer>"]),
                dataset=dataset,
                output_dir=root,
                checkpoint_path=checkpoint,
                checkpoint_sha256=file_sha256(checkpoint),
                seed=1,
                start=0,
                chunk_size=3,
            )
            self.assertEqual(result["sample_count"], 3)
            self.assertEqual(len(_read_jsonl(paths["predictions"])), 3)

    def test_complete_chunk_is_reused_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"prefix checkpoint")
            dataset = _items(1)
            first = run_drop_chunk(
                model=_Generator(["<answer>0</answer>"]),
                dataset=dataset,
                output_dir=root,
                checkpoint_path=checkpoint,
                checkpoint_sha256=file_sha256(checkpoint),
                seed=1,
                start=0,
                chunk_size=1,
            )
            reused = run_drop_chunk(
                model=None,
                dataset=dataset,
                output_dir=root,
                checkpoint_path=checkpoint,
                checkpoint_sha256=file_sha256(checkpoint),
                seed=1,
                start=0,
                chunk_size=1,
            )
            self.assertFalse(first["reused"])
            self.assertTrue(reused["reused"])

    def test_repeated_official_query_id_is_not_collapsed_in_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"prefix checkpoint")
            dataset = _items(2)
            dataset[1]["id"] = dataset[0]["id"]
            result = run_drop_chunk(
                model=_Generator(["<answer>0</answer>", "<answer>1</answer>"]),
                dataset=dataset,
                output_dir=root,
                checkpoint_path=checkpoint,
                checkpoint_sha256=file_sha256(checkpoint),
                seed=1,
                start=0,
                chunk_size=2,
            )
            self.assertEqual(result["progress"]["recorded_samples"], 2)

    def test_missing_or_tampered_chunk_blocks_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"prefix checkpoint")
            dataset = _items(2)
            with self.assertRaisesRegex(RuntimeError, "missing complete"):
                merge_drop_chunks(
                    dataset=dataset,
                    output_dir=root,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=file_sha256(checkpoint),
                    seed=1,
                    chunk_size=2,
                )
            run_drop_chunk(
                model=_Generator(["<answer>0</answer>", "<answer>1</answer>"]),
                dataset=dataset,
                output_dir=root,
                checkpoint_path=checkpoint,
                checkpoint_sha256=file_sha256(checkpoint),
                seed=1,
                start=0,
                chunk_size=2,
            )
            paths = chunk_paths(root, 0, 2)
            paths["predictions"].write_text(paths["predictions"].read_text("utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                merge_drop_chunks(
                    dataset=dataset,
                    output_dir=root,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=file_sha256(checkpoint),
                    seed=1,
                    chunk_size=2,
                )


if __name__ == "__main__":
    unittest.main()
