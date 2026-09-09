"""Lightweight tests for F0/F1 checkpoint cross-composition evaluation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from skillopt.behavior_compression.new_task_transfer.v1.cross_composition import (
    ALL_COMPOSITIONS,
    DEFAULT_COMPOSITIONS,
    RAW_SHARED_F0_TASK,
    RAW_SHARED_F1_TASK,
    TRANSFERRED_SHARED_F0_TASK,
    TRANSFERRED_SHARED_F1_TASK,
    checkpoint_geometry,
    compose_prefix_blocks,
    deterministic_subset,
    load_frozen_transfer_checkpoint,
    run_cross_composition_evaluation,
    validate_checkpoint_pair,
)
from skillopt.behavior_compression.new_task_transfer.v1.model import (
    TransferPrefixParameters,
)
from skillopt.behavior_compression.new_task_transfer.v1.train_state import (
    PROTOCOL_VERSION as TRAIN_CHECKPOINT_PROTOCOL,
)


HIDDEN_SIZE = 5
SAMPLE_HASH = "a" * 64


def _checkpoint(condition: str, shared: torch.Tensor, task: torch.Tensor) -> dict:
    return {
        "protocol_version": TRAIN_CHECKPOINT_PROTOCOL,
        "task": "drop",
        "condition": condition,
        "seed": 1,
        "optimizer_step": 150,
        "sample_order_hash": SAMPLE_HASH,
        "prefix_state": {
            "protocol_version": "new_task_transfer_v1",
            "model_name": "mock-qwen",
            "task_name": "drop",
            "condition": condition.upper(),
            "prefix_length": 16,
            "effective_prefix_length": 32,
            "use_residual_reparameterization": False,
            "prefix_tensors": {
                "shared_prefix_embeddings": shared,
                "task_prefix_embeddings": task,
            },
            "source_checkpoint": (
                {"tensor_sha256": "fixture-source"} if condition == "f1" else None
            ),
        },
    }


class _CrossModel:
    def __init__(self) -> None:
        self.torch = torch
        self.model = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.model.requires_grad_(False)
        initial = torch.zeros(16, HIDDEN_SIZE)
        self.prefix_parameters = TransferPrefixParameters(
            initial,
            initial.clone(),
            condition="f0",
        )
        self.condition = "F0"
        self.use_residual_reparameterization = False

    def active_prefix_embeddings(self):
        return self.prefix_parameters.active_prefix_embeddings()

    def shared_prefix_tensor(self):
        return self.prefix_parameters.shared_prefix_tensor()

    def task_prefix_tensor(self):
        return self.prefix_parameters.task_prefix_tensor()


class CrossCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        base = torch.arange(16 * HIDDEN_SIZE, dtype=torch.float32).reshape(
            16, HIDDEN_SIZE
        )
        self.f0_shared = base + 1.0
        self.f0_task = base + 101.0
        self.f1_shared = base + 201.0
        self.f1_task = base + 301.0

    def _write_pair(self, root: Path):
        f0_path = root / "f0.pt"
        f1_path = root / "f1.pt"
        torch.save(_checkpoint("f0", self.f0_shared, self.f0_task), f0_path)
        torch.save(_checkpoint("f1", self.f1_shared, self.f1_task), f1_path)
        f0 = load_frozen_transfer_checkpoint(
            f0_path,
            expected_condition="f0",
            expected_task="drop",
            expected_seed=1,
            torch_module=torch,
        )
        f1 = load_frozen_transfer_checkpoint(
            f1_path,
            expected_condition="f1",
            expected_task="drop",
            expected_seed=1,
            torch_module=torch,
        )
        return f0, f1

    def test_loads_strict_final_pair_and_reports_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            f0, f1 = self._write_pair(Path(temporary))
            validate_checkpoint_pair(f0, f1)
            self.assertEqual(f0.optimizer_step, 150)
            self.assertEqual(f0.sample_order_hash, f1.sample_order_hash)
            geometry = checkpoint_geometry(f0, f1, torch_module=torch)
            self.assertIn("shared_raw_f0_vs_transferred_f1", geometry)
            self.assertGreater(
                geometry["final_task_f0_vs_f1"]["delta_norm"], 0.0
            )

    def test_constructs_two_native_and_two_crossed_prefixes(self) -> None:
        expected = {
            RAW_SHARED_F0_TASK: (self.f0_shared, self.f0_task),
            TRANSFERRED_SHARED_F1_TASK: (self.f1_shared, self.f1_task),
            TRANSFERRED_SHARED_F0_TASK: (self.f1_shared, self.f0_task),
            RAW_SHARED_F1_TASK: (self.f0_shared, self.f1_task),
        }
        with tempfile.TemporaryDirectory() as temporary:
            f0, f1 = self._write_pair(Path(temporary))
            for name, (expected_shared, expected_task) in expected.items():
                with self.subTest(composition=name):
                    shared, task, record = compose_prefix_blocks(f0, f1, name)
                    self.assertTrue(torch.equal(shared, expected_shared))
                    self.assertTrue(torch.equal(task, expected_task))
                    self.assertEqual(record["effective_prefix_length"], 32)

    def test_rejects_mismatched_pair_and_malformed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            f0, f1 = self._write_pair(root)
            object.__setattr__(f1, "sample_order_hash", "b" * 64)
            with self.assertRaisesRegex(RuntimeError, "sample_order_hash"):
                validate_checkpoint_pair(f0, f1)

            malformed = _checkpoint("f0", self.f0_shared[:15], self.f0_task)
            bad_path = root / "bad.pt"
            torch.save(malformed, bad_path)
            with self.assertRaisesRegex(ValueError, "shape"):
                load_frozen_transfer_checkpoint(
                    bad_path,
                    expected_condition="f0",
                    expected_task="drop",
                    expected_seed=1,
                    torch_module=torch,
                )

    def test_evaluates_all_compositions_and_publishes_complete_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            f0, f1 = self._write_pair(root)
            model = _CrossModel()
            observed: list[torch.Tensor] = []

            def evaluator(**kwargs):
                observed.append(kwargs["model"].active_prefix_embeddings().detach().clone())
                path = Path(kwargs["predictions_path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"sample_id":"drop:1"}\n', encoding="utf-8")
                return {
                    "complete": True,
                    "metrics": {"em": 0.5, "f1": 0.75},
                    "num_samples": len(kwargs["dataset"]),
                    "generation": {"enable_thinking": False},
                }

            output = root / "cross"
            summary = run_cross_composition_evaluation(
                model=model,
                f0=f0,
                f1=f1,
                dataset=[{"id": "drop:1"}],
                split="validation",
                output_dir=output,
                compositions=ALL_COMPOSITIONS,
                evaluator=evaluator,
                dataset_id_hash="fixture-hash",
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(tuple(summary["compositions"]), ALL_COMPOSITIONS)
            self.assertEqual(len(observed), 4)
            self.assertTrue((output / "cross_composition_summary.json").is_file())
            for name in ALL_COMPOSITIONS:
                self.assertTrue((output / "predictions" / f"{name}.jsonl").is_file())
            self.assertTrue(torch.equal(observed[0][:16], self.f0_shared))
            self.assertTrue(torch.equal(observed[2][:16], self.f1_shared))
            self.assertTrue(torch.equal(observed[2][16:], self.f0_task))
            on_disk = json.loads(
                (output / "cross_composition_summary.json").read_text("utf-8")
            )
            self.assertEqual(on_disk["dataset_id_hash"], "fixture-hash")

    def test_default_evaluates_only_missing_crossed_pairs(self) -> None:
        self.assertEqual(
            DEFAULT_COMPOSITIONS,
            (TRANSFERRED_SHARED_F0_TASK, RAW_SHARED_F1_TASK),
        )

    def test_failure_does_not_publish_complete_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            f0, f1 = self._write_pair(root)

            def evaluator(**_kwargs):
                raise RuntimeError("generation failed")

            output = root / "cross"
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                run_cross_composition_evaluation(
                    model=_CrossModel(),
                    f0=f0,
                    f1=f1,
                    dataset=[{"id": "drop:1"}],
                    split="validation",
                    output_dir=output,
                    compositions=[RAW_SHARED_F0_TASK],
                    evaluator=evaluator,
                )
            self.assertFalse((output / "cross_composition_summary.json").exists())

    def test_deterministic_subset_is_reproducible_and_order_preserving(self) -> None:
        items = [{"id": f"drop:{index}"} for index in range(20)]
        first, first_hash = deterministic_subset(
            items, task="drop", split="validation", seed=1, max_samples=7
        )
        second, second_hash = deterministic_subset(
            items, task="drop", split="validation", seed=1, max_samples=7
        )
        self.assertEqual(first, second)
        self.assertEqual(first_hash, second_hash)
        indices = [int(item["id"].split(":")[1]) for item in first]
        self.assertEqual(indices, sorted(indices))


if __name__ == "__main__":
    unittest.main()
