"""Directed mock tests for the stage-3A minimal training core."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from skillopt.behavior_compression.new_task_transfer.v1.model import (
    CONDITION_F0,
    CONDITION_F1,
    CONDITION_I32,
    TransferPrefixParameters,
)
from skillopt.behavior_compression.new_task_transfer.v1.train_core import (
    MinimalTrainConfig,
    build_deterministic_batch_schedule,
    run_minimal_training,
)


class _MockDataset:
    def __init__(self, task: str, size: int = 7) -> None:
        self.items = [
            {"id": f"{task}:{index}", "target": index + 1}
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

    def __init__(self, condition: str, task: str = "chartqa") -> None:
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
        self.condition = condition
        self.task_name = task
        self.use_residual_reparameterization = False
        self.forward_calls = 0

    def active_prefix_embeddings(self):
        return self.prefix_parameters.active_prefix_embeddings()

    def trainable_parameters(self):
        return self.prefix_parameters.trainable_parameters()

    def validate_invariants(self) -> None:
        self.prefix_parameters.validate()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("mock base model is not frozen")

    def forward(self, batch: dict):
        self.forward_calls += 1
        supervised = batch["labels"][batch["labels"] != -100].float().mean()
        target = supervised / 100.0
        losses = [
            (parameter.float() - target).square().mean()
            for parameter in self.trainable_parameters()
        ]
        return SimpleNamespace(loss=sum(losses))


def _config(
    condition: str,
    *,
    accumulation: int = 1,
    seed: int = 11,
) -> MinimalTrainConfig:
    return MinimalTrainConfig(
        task="chartqa",
        condition=condition.lower(),
        seed=seed,
        source_checkpoint="fixture.pt" if condition == CONDITION_F1 else "",
        output_dir="unused-by-stage3a-mock",
        max_optimizer_steps=2,
        learning_rate=1e-3,
        gradient_accumulation_steps=accumulation,
        batch_size=1,
    )


class MinimalTrainingCoreTests(unittest.TestCase):
    def test_mock_training_runs_exactly_two_optimizer_steps(self) -> None:
        model = _MockTransferModel(CONDITION_F0)
        result = run_minimal_training(
            config=_config(CONDITION_F0),
            model=model,
            train_dataset=_MockDataset("chartqa"),
            collator=_collate,
        )
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(model.forward_calls, 2)

    def test_accumulation_consumes_microbatches_before_each_step(self) -> None:
        model = _MockTransferModel(CONDITION_F0)
        result = run_minimal_training(
            config=_config(CONDITION_F0, accumulation=3),
            model=model,
            train_dataset=_MockDataset("chartqa"),
            collator=_collate,
        )
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["microbatches_consumed"], 6)
        self.assertEqual(model.forward_calls, 6)

    def test_each_condition_updates_only_its_allowed_prefix(self) -> None:
        for condition in (CONDITION_F0, CONDITION_F1, CONDITION_I32):
            with self.subTest(condition=condition):
                model = _MockTransferModel(condition)
                shared_before = (
                    model.prefix_parameters.shared_prefix_tensor().detach().clone()
                )
                task_before = model.prefix_parameters.task_prefix_tensor().detach().clone()
                run_minimal_training(
                    config=_config(condition),
                    model=model,
                    train_dataset=_MockDataset("chartqa"),
                    collator=_collate,
                )
                shared_after = model.prefix_parameters.shared_prefix_tensor().detach()
                task_after = model.prefix_parameters.task_prefix_tensor().detach()
                if condition == CONDITION_I32:
                    self.assertFalse(torch.equal(shared_before, shared_after))
                else:
                    self.assertTrue(torch.equal(shared_before, shared_after))
                self.assertFalse(torch.equal(task_before, task_after))

    def test_sample_order_hash_is_condition_independent(self) -> None:
        sample_ids = [f"chartqa:{index}" for index in range(7)]
        hashes = []
        for _condition in (CONDITION_F0, CONDITION_F1, CONDITION_I32):
            schedule = build_deterministic_batch_schedule(
                sample_ids,
                task="chartqa",
                seed=11,
                batch_size=2,
                gradient_accumulation_steps=2,
                max_optimizer_steps=2,
            )
            hashes.append(schedule.sample_order_hash)
        self.assertEqual(len(set(hashes)), 1)


if __name__ == "__main__":
    unittest.main()
