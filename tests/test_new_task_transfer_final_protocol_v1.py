"""Directed mock tests for stage-3C2 final-only orchestration."""
from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import tensor_sha256
from skillopt.behavior_compression.new_task_transfer.v1.evaluation import (
    evaluate_transfer_predictions,
)
from skillopt.behavior_compression.new_task_transfer.v1.final_protocol import (
    run_final_only_experiment,
    run_training_to_final_checkpoint,
)
from skillopt.behavior_compression.new_task_transfer.v1.train_core import (
    MinimalTrainConfig,
)


class _Tokenizer:
    chat_template = "mock-qwen"

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return "rendered-drop-prompt"


class _MockFinalModel:
    torch = torch

    def __init__(self, task: str, condition: str) -> None:
        self.task_name = task
        self.condition = condition.upper()
        self.shared = torch.full((16, 2), 0.3 if condition == "f1" else 0.1)
        self.task = torch.full((16, 2), 0.2)
        self.tokenizer = _Tokenizer()
        self.model = SimpleNamespace(eval=lambda: None)

    def shared_prefix_tensor(self):
        return self.shared

    def state_dict(self) -> dict:
        return {
            "task": self.task_name,
            "condition": self.condition,
            "shared": self.shared.clone(),
            "task_prefix": self.task.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.shared.copy_(state["shared"])
        self.task.copy_(state["task_prefix"])

    def generate_from_messages(self, messages, **kwargs) -> str:
        del messages, kwargs
        return "<answer>blue</answer>"

    def generate_from_prompt(self, prompt, **kwargs) -> str:
        del prompt, kwargs
        return "<answer>12</answer>"


class _TrainDataset:
    def __init__(self, task: str) -> None:
        self.items = [{"id": f"{task}:train:{index}"} for index in range(4)]


class _FakeTrainingRunner:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        *,
        config,
        model,
        train_dataset,
        collator,
        resume,
        resume_checkpoint,
        checkpoint_every_steps,
        optimizer_step_callback,
    ) -> dict:
        del train_dataset, collator, resume, resume_checkpoint, checkpoint_every_steps
        self.calls += 1
        output = Path(config.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        sample_hash = hashlib.sha256(
            f"{config.task}:{config.seed}:fixed-order".encode()
        ).hexdigest()
        source = None
        if config.condition == "f1":
            source = {
                "checkpoint_path": str(Path(config.source_checkpoint).resolve()),
                "checkpoint_sha256": "source-file-hash",
                "state_key": "model.shared_prefix_embeddings",
                "shared_tensor_sha256": tensor_sha256(model.shared_prefix_tensor()),
            }
        audit = {
            "status": "passed",
            "task": config.task,
            "condition": config.condition,
            "seed": config.seed,
            "prefix_total_length": 32,
            "base_model": {"fully_frozen": True},
            "residual_enabled": False,
            "sample_order_hash": sample_hash,
            "f1_source": source,
            "initialization_token_info_reference": {
                "shared_behavior": {"path": "shared_behavior.md"}
            },
        }
        sanity = {
            "status": "passed",
            "task": config.task,
            "condition": config.condition,
            "seed": config.seed,
            "checks": {"mock": {"passed": True, "evidence": "mock"}},
        }
        (output / "step0_audit.json").write_text(json.dumps(audit), "utf-8")
        (output / "transfer_sanity.json").write_text(json.dumps(sanity), "utf-8")
        last_metric = None
        for step in range(1, 151):
            last_metric = {
                "optimizer_step": step,
                "loss": 1.0 / step,
                "microbatches_consumed": step,
                "step_elapsed_seconds": 0.001,
            }
            optimizer_step_callback(last_metric)
        checkpoint = {
            "task": config.task,
            "condition": config.condition,
            "seed": config.seed,
            "optimizer_step": 150,
            "sample_order_hash": sample_hash,
            "prefix_state": model.state_dict(),
            "last_train_metric": last_metric,
        }
        torch.save(checkpoint, output / "latest.pt")
        torch.save(checkpoint, output / "final.pt")
        return {"optimizer_steps": 150, "sample_order_hash": sample_hash}


def _config(output: Path, task: str, condition: str, *, steps: int = 150):
    source = output.parent.parent / "source.pt"
    if condition == "f1":
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"mock source")
    return MinimalTrainConfig(
        task=task,
        condition=condition,
        seed=1,
        source_checkpoint=str(source) if condition == "f1" else "",
        output_dir=str(output),
        max_optimizer_steps=steps,
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        batch_size=1,
    )


def _chart_splits() -> dict[str, list[dict]]:
    return {
        "validation": [
            {
                "id": "chartqa:val:0",
                "split": "validation",
                "source_type": "human",
                "question": "Which color?",
                "answer": "blue",
                "image_path": "/mock/val.png",
            }
        ],
        "test": [
            {
                "id": "chartqa:test:0",
                "split": "test",
                "source_type": "human",
                "question": "Which color?",
                "answer": "blue",
                "image_path": "/mock/test.png",
            }
        ],
    }


def _drop_splits() -> dict[str, list[dict]]:
    answer = {"number": "12", "date": {}, "spans": []}
    return {
        "validation": [
            {
                "id": "drop:val:0",
                "split": "validation",
                "passage": "The score was 12.",
                "question": "What was the score?",
                "candidate_answers": [answer],
                "answer_type": "number",
            }
        ]
    }


def _run(
    *,
    output: Path,
    task: str,
    condition: str = "f0",
    runner: _FakeTrainingRunner | None = None,
    evaluator=evaluate_transfer_predictions,
    resume: bool = False,
):
    runner = runner or _FakeTrainingRunner()
    model = _MockFinalModel(task, condition)
    summary = run_final_only_experiment(
        config=_config(output, task, condition),
        model=model,
        train_dataset=_TrainDataset(task),
        collator=lambda examples: examples,
        evaluation_splits=_chart_splits() if task == "chartqa" else _drop_splits(),
        resume=resume,
        training_runner=runner,
        evaluator=evaluator,
    )
    return summary, runner


class FinalOnlyProtocolTests(unittest.TestCase):
    def test_training_only_stops_after_final_checkpoint_without_eval_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "chartqa" / "f0" / "seed1"
            runner = _FakeTrainingRunner()
            result = run_training_to_final_checkpoint(
                config=_config(output, "chartqa", "f0"),
                model=_MockFinalModel("chartqa", "f0"),
                train_dataset=_TrainDataset("chartqa"),
                collator=lambda examples: examples,
                training_runner=runner,
            )
            self.assertEqual(result["status"], "training_complete")
            self.assertEqual(result["optimizer_steps"], 150)
            self.assertTrue((output / "final.pt").is_file())
            self.assertFalse((output / "eval_metrics.jsonl").exists())
            self.assertFalse((output / "predictions").exists())
            self.assertFalse((output / "summary.json").exists())

    def test_chartqa_uses_final_validation_and_test_with_test_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "chartqa" / "f0" / "seed1"
            summary, _ = _run(output=output, task="chartqa")
            eval_rows = [
                json.loads(line)
                for line in (output / "eval_metrics.jsonl").read_text("utf-8").splitlines()
            ]
            self.assertEqual([row["split"] for row in eval_rows], ["validation", "test"])
            self.assertTrue(all(row["optimizer_step"] == 150 for row in eval_rows))
            self.assertEqual(summary["report_split"], "test")
            self.assertEqual(summary["metrics"], {"relaxed_accuracy": 1.0})
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["checkpoint"]["selection"], "final_only_no_validation_best")
            self.assertEqual(
                len((output / "train_metrics.jsonl").read_text("utf-8").splitlines()),
                150,
            )

    def test_drop_uses_only_step150_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "drop" / "f0" / "seed1"
            summary, _ = _run(output=output, task="drop")
            eval_rows = [
                json.loads(line)
                for line in (output / "eval_metrics.jsonl").read_text("utf-8").splitlines()
            ]
            self.assertEqual([row["split"] for row in eval_rows], ["validation"])
            self.assertEqual(summary["report_split"], "validation")
            self.assertEqual(summary["metrics"], {"em": 1.0, "f1": 1.0})
            self.assertNotIn("best", " ".join(path.name for path in output.iterdir()))

    def test_protocol_rejects_any_training_length_other_than_150(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "chartqa" / "f0" / "seed1"
            with self.assertRaisesRegex(ValueError, "exactly 150"):
                run_final_only_experiment(
                    config=_config(output, "chartqa", "f0", steps=149),
                    model=_MockFinalModel("chartqa", "f0"),
                    train_dataset=_TrainDataset("chartqa"),
                    collator=lambda examples: examples,
                    evaluation_splits=_chart_splits(),
                )

    def test_resume_complete_run_does_not_duplicate_metrics_or_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "chartqa" / "f0" / "seed1"
            runner = _FakeTrainingRunner()
            summary, _ = _run(output=output, task="chartqa", runner=runner)
            before = {
                name: (output / name).read_bytes()
                for name in ("train_metrics.jsonl", "eval_metrics.jsonl", "summary.json")
            }
            predictions_before = {
                path.name: path.read_bytes() for path in (output / "predictions").iterdir()
            }
            resumed, _ = _run(
                output=output,
                task="chartqa",
                runner=runner,
                resume=True,
            )
            self.assertEqual(runner.calls, 1)
            self.assertEqual(summary, resumed)
            self.assertEqual(
                before,
                {name: (output / name).read_bytes() for name in before},
            )
            self.assertEqual(
                predictions_before,
                {path.name: path.read_bytes() for path in (output / "predictions").iterdir()},
            )

    def test_summary_appears_only_after_all_formal_evaluations_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "chartqa" / "f0" / "seed1"

            def fail_test(**kwargs):
                if kwargs["split"] == "test":
                    raise RuntimeError("intentional test failure")
                return evaluate_transfer_predictions(**kwargs)

            with self.assertRaisesRegex(RuntimeError, "intentional test failure"):
                _run(output=output, task="chartqa", evaluator=fail_test)
            self.assertFalse((output / "summary.json").exists())
            summary, _ = _run(output=output, task="chartqa", resume=True)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(
                len((output / "eval_metrics.jsonl").read_text("utf-8").splitlines()),
                2,
            )

    def test_resume_promotes_completed_latest_checkpoint_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "drop" / "f0" / "seed1"
            runner = _FakeTrainingRunner()
            _run(output=output, task="drop", runner=runner)
            (output / "final.pt").unlink()
            (output / "summary.json").unlink()
            summary, _ = _run(
                output=output,
                task="drop",
                runner=runner,
                resume=True,
            )
            self.assertEqual(runner.calls, 1)
            self.assertTrue((output / "final.pt").is_file())
            self.assertEqual(summary["status"], "complete")

    def test_three_condition_directories_are_isolated_with_equal_sample_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "chartqa"
            summaries = {}
            for condition in ("f0", "f1", "i32"):
                output = root / condition / "seed1"
                summaries[condition], _ = _run(
                    output=output,
                    task="chartqa",
                    condition=condition,
                )
                self.assertTrue((output / "predictions" / "test.jsonl").is_file())
            self.assertEqual(
                len({summary["sample_order_hash"] for summary in summaries.values()}),
                1,
            )
            self.assertEqual(
                {key: value["trainable_prefix_tokens"] for key, value in summaries.items()},
                {"f0": 16, "f1": 16, "i32": 32},
            )
            self.assertEqual(len(list(root.glob("*/seed1/summary.json"))), 3)

    def test_evaluation_restores_python_and_torch_rng_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "drop" / "f0" / "seed1"
            random.seed(123)
            torch.manual_seed(123)

            def consuming_evaluator(**kwargs):
                random.random()
                torch.rand(3)
                return evaluate_transfer_predictions(**kwargs)

            python_before = random.getstate()
            torch_before = torch.get_rng_state().clone()
            _run(output=output, task="drop", evaluator=consuming_evaluator)
            self.assertEqual(random.getstate(), python_before)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))


if __name__ == "__main__":
    unittest.main()
