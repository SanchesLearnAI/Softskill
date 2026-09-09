"""Directed mock tests for stage-3B audit, sanity, and resume state."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import (
    SHARED_CHECKPOINT_STATE_KEY,
    file_sha256,
    tensor_sha256,
)
from skillopt.behavior_compression.new_task_transfer.v1.model import (
    CONDITION_F0,
    CONDITION_F1,
    CONDITION_I32,
    TransferPrefixParameters,
)
from skillopt.behavior_compression.new_task_transfer.v1.train_core import (
    MinimalTrainConfig,
)
from skillopt.behavior_compression.new_task_transfer.v1.train_state import (
    run_stateful_training,
)


class _InterruptedAfterCheckpoint(RuntimeError):
    pass


class _MockDataset:
    def __init__(self, task: str = "chartqa", size: int = 7, suffix: str = "") -> None:
        self.items = [
            {"id": f"{task}:{index}{suffix}", "target": index + 1}
            for index in range(size)
        ]

    def __getitem__(self, index: int) -> dict:
        target = int(self.items[index]["target"])
        return {
            "input_ids": [1, target],
            "attention_mask": [1, 1],
            "labels": [-100, target],
        }


def _collate(examples: list[dict]) -> dict:
    return {
        key: torch.tensor([example[key] for example in examples], dtype=torch.long)
        for key in ("input_ids", "attention_mask", "labels")
    }


class _MockTransferModel:
    torch = torch
    device = torch.device("cpu")

    def __init__(
        self,
        condition: str,
        *,
        task: str = "chartqa",
        source_path: Path | None = None,
    ) -> None:
        raw_shared = torch.full((16, 4), 0.10)
        task_block = torch.full((16, 4), 0.20)
        transferred = torch.full((16, 4), 0.30)
        self.prefix_parameters = TransferPrefixParameters(
            raw_shared,
            task_block,
            condition=condition,
            transferred_shared_block=(
                transferred if condition == CONDITION_F1 else None
            ),
        )
        self.model = torch.nn.Linear(1, 1, bias=False)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.config = SimpleNamespace(_commit_hash="mock")
        self.model_name = "mock-frozen-base"
        self.condition = condition
        self.task_name = task
        self.use_residual_reparameterization = False
        self.initialization_audit = {
            "shared_behavior": {
                "path": "shared_behavior.md",
                "selected_token_ids": list(range(16)),
            },
            "task_behavior": {
                "path": f"{task}_behavior.md",
                "selected_token_ids": list(range(16, 32)),
            },
        }
        self.source_metadata = {}
        if condition == CONDITION_F1:
            if source_path is None:
                raise ValueError("mock F1 requires source_path")
            self.source_metadata = {
                "checkpoint_path": str(source_path.resolve()),
                "checkpoint_sha256": file_sha256(source_path),
                "state_key": SHARED_CHECKPOINT_STATE_KEY,
                "tensor_sha256": tensor_sha256(
                    self.prefix_parameters.shared_prefix_tensor()
                ),
            }

    def active_prefix_embeddings(self):
        return self.prefix_parameters.active_prefix_embeddings()

    def shared_prefix_tensor(self):
        return self.prefix_parameters.shared_prefix_tensor()

    def trainable_parameters(self):
        return self.prefix_parameters.trainable_parameters()

    def validate_invariants(self) -> None:
        self.prefix_parameters.validate()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("mock base model is not frozen")

    def forward(self, batch: dict):
        supervised = batch["labels"][batch["labels"] != -100].float().mean()
        target = supervised / 100.0
        losses = [
            (parameter.float() - target).square().mean()
            for parameter in self.trainable_parameters()
        ]
        return SimpleNamespace(loss=sum(losses))

    def state_dict(self) -> dict:
        return {
            "protocol_version": "mock_new_task_transfer_v1",
            "model_name": self.model_name,
            "task_name": self.task_name,
            "condition": self.condition,
            "prefix_tensors": {
                key: value.detach().cpu().clone()
                for key, value in self.prefix_parameters.export_tensors().items()
            },
            "initialization_audit": self.initialization_audit,
            "source_checkpoint": self.source_metadata or None,
        }

    def load_state_dict(self, state: dict) -> None:
        for key, expected in (
            ("model_name", self.model_name),
            ("task_name", self.task_name),
            ("condition", self.condition),
        ):
            if state.get(key) != expected:
                raise ValueError(f"mock {key} mismatch")
        self.prefix_parameters.load_exported_tensors(state["prefix_tensors"])
        self.initialization_audit = dict(state.get("initialization_audit") or {})
        self.source_metadata = dict(state.get("source_checkpoint") or {})


def _config(
    output_dir: Path,
    condition: str,
    *,
    source_path: Path | None = None,
    steps: int = 3,
    seed: int = 11,
    learning_rate: float = 1e-3,
) -> MinimalTrainConfig:
    return MinimalTrainConfig(
        task="chartqa",
        condition=condition.lower(),
        seed=seed,
        source_checkpoint=(str(source_path) if source_path else ""),
        output_dir=str(output_dir),
        max_optimizer_steps=steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=2,
        batch_size=1,
    )


def _interrupt_at_one(step: int) -> None:
    if step == 1:
        raise _InterruptedAfterCheckpoint("simulated interruption")


class StatefulTrainingTests(unittest.TestCase):
    def _source_file(self, root: Path) -> Path:
        path = root / "source.pt"
        path.write_bytes(b"mock-shared-source")
        return path

    def test_step0_audit_contains_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            config = _config(output, CONDITION_F0, steps=1)
            run_stateful_training(
                config=config,
                model=_MockTransferModel(CONDITION_F0),
                train_dataset=_MockDataset(),
                collator=_collate,
            )
            audit = json.loads((output / "step0_audit.json").read_text("utf-8"))
            required = {
                "task",
                "condition",
                "seed",
                "prefix_total_length",
                "prefix_parameters",
                "base_model",
                "residual_enabled",
                "optimizer",
                "sample_order_hash",
                "f1_source",
                "initialization_token_info_reference",
            }
            self.assertTrue(required.issubset(audit))
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["prefix_total_length"], 32)

    def test_all_three_conditions_pass_first_step_sanity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source_file(root)
            for condition in (CONDITION_F0, CONDITION_F1, CONDITION_I32):
                with self.subTest(condition=condition):
                    output = root / condition.lower()
                    model = _MockTransferModel(
                        condition,
                        source_path=source if condition == CONDITION_F1 else None,
                    )
                    run_stateful_training(
                        config=_config(
                            output,
                            condition,
                            source_path=source if condition == CONDITION_F1 else None,
                            steps=1,
                        ),
                        model=model,
                        train_dataset=_MockDataset(),
                        collator=_collate,
                    )
                    sanity = json.loads(
                        (output / "transfer_sanity.json").read_text("utf-8")
                    )
                    self.assertEqual(sanity["status"], "passed")
                    self.assertTrue(
                        all(item["passed"] for item in sanity["checks"].values())
                    )

    def test_frozen_prefix_mutation_fails_sanity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            model = _MockTransferModel(CONDITION_F0)

            def mutate_frozen(current_model) -> None:
                with torch.no_grad():
                    current_model.prefix_parameters.shared_prefix.add_(1.0)

            with self.assertRaisesRegex(RuntimeError, "sanity failed"):
                run_stateful_training(
                    config=_config(output, CONDITION_F0, steps=1),
                    model=model,
                    train_dataset=_MockDataset(),
                    collator=_collate,
                    first_step_post_optimizer_hook=mutate_frozen,
                )
            sanity = json.loads(
                (output / "transfer_sanity.json").read_text("utf-8")
            )
            self.assertEqual(sanity["status"], "failed")
            self.assertFalse(
                sanity["checks"]["frozen_prefix_unchanged"]["passed"]
            )

    def test_one_step_checkpoint_resumes_to_three(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            config = _config(output, CONDITION_F0, steps=3)
            with self.assertRaises(_InterruptedAfterCheckpoint):
                run_stateful_training(
                    config=config,
                    model=_MockTransferModel(CONDITION_F0),
                    train_dataset=_MockDataset(),
                    collator=_collate,
                    after_checkpoint_hook=_interrupt_at_one,
                )
            step_one = torch.load(output / "latest.pt", weights_only=False)
            self.assertEqual(step_one["optimizer_step"], 1)
            result = run_stateful_training(
                config=config,
                model=_MockTransferModel(CONDITION_F0),
                train_dataset=_MockDataset(),
                collator=_collate,
                resume=True,
            )
            final = torch.load(output / "final.pt", weights_only=False)
            self.assertEqual(result["start_optimizer_step"], 1)
            self.assertEqual(final["optimizer_step"], 3)

    def test_resumed_parameters_match_continuous_three_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous_model = _MockTransferModel(CONDITION_I32)
            run_stateful_training(
                config=_config(root / "continuous", CONDITION_I32, steps=3),
                model=continuous_model,
                train_dataset=_MockDataset(),
                collator=_collate,
            )
            interrupted_config = _config(root / "resumed", CONDITION_I32, steps=3)
            with self.assertRaises(_InterruptedAfterCheckpoint):
                run_stateful_training(
                    config=interrupted_config,
                    model=_MockTransferModel(CONDITION_I32),
                    train_dataset=_MockDataset(),
                    collator=_collate,
                    after_checkpoint_hook=_interrupt_at_one,
                )
            resumed_model = _MockTransferModel(CONDITION_I32)
            run_stateful_training(
                config=interrupted_config,
                model=resumed_model,
                train_dataset=_MockDataset(),
                collator=_collate,
                resume=True,
            )
            self.assertTrue(
                torch.equal(
                    continuous_model.active_prefix_embeddings(),
                    resumed_model.active_prefix_embeddings(),
                )
            )

    def test_resume_rejects_config_or_sample_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            config = _config(output, CONDITION_F0, steps=3)
            with self.assertRaises(_InterruptedAfterCheckpoint):
                run_stateful_training(
                    config=config,
                    model=_MockTransferModel(CONDITION_F0),
                    train_dataset=_MockDataset(),
                    collator=_collate,
                    after_checkpoint_hook=_interrupt_at_one,
                )
            with self.assertRaisesRegex(RuntimeError, "config"):
                run_stateful_training(
                    config=_config(
                        output,
                        CONDITION_F0,
                        steps=3,
                        learning_rate=2e-3,
                    ),
                    model=_MockTransferModel(CONDITION_F0),
                    train_dataset=_MockDataset(),
                    collator=_collate,
                    resume=True,
                )
            with self.assertRaisesRegex(RuntimeError, "sample order hash"):
                run_stateful_training(
                    config=config,
                    model=_MockTransferModel(CONDITION_F0),
                    train_dataset=_MockDataset(suffix=":changed"),
                    collator=_collate,
                    resume=True,
                )

    def test_checkpoint_contains_prefix_not_base_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run_stateful_training(
                config=_config(output, CONDITION_F0, steps=1),
                model=_MockTransferModel(CONDITION_F0),
                train_dataset=_MockDataset(),
                collator=_collate,
            )
            checkpoint = torch.load(output / "final.pt", weights_only=False)
            self.assertIn("prefix_state", checkpoint)
            self.assertNotIn("base_model", checkpoint)
            self.assertNotIn("base_model_state", checkpoint)
            self.assertNotIn("model_state_dict", checkpoint)
            self.assertEqual(
                set(checkpoint["prefix_state"]["prefix_tensors"]),
                {"shared_prefix_embeddings", "task_prefix_embeddings"},
            )

    def test_nonempty_output_directory_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-empty"):
                run_stateful_training(
                    config=_config(output, CONDITION_F0, steps=1),
                    model=_MockTransferModel(CONDITION_F0),
                    train_dataset=_MockDataset(),
                    collator=_collate,
                )
            self.assertEqual(marker.read_text("utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
