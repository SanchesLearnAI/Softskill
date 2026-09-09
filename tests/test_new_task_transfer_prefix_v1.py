"""Lightweight stage-2 tests for target-only F0/F1/I32 prefixes."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from skillopt.behavior_compression.new_task_transfer.v1.checkpoint import (
    SHARED_CHECKPOINT_STATE_KEY,
    SOURCE_CHECKPOINT_CANDIDATES,
    extract_shared_prefix,
)
from skillopt.behavior_compression.new_task_transfer.v1.model import (
    CONDITION_F0,
    CONDITION_F1,
    CONDITION_I32,
    EFFECTIVE_PREFIX_LENGTH,
    SHARED_BEHAVIOR_PATH,
    SHARED_TOKEN_AUDIT_PATH,
    TASK_BEHAVIOR_PATHS,
    TASK_TOKEN_AUDIT_PATH,
    TargetOnlyTransferSoftPrefixVisionLM,
    TransferPrefixParameters,
    embed_task_behavior_blocks,
)


HIDDEN_SIZE = 8
REAL_SHARED_CHECKPOINT = os.environ.get("SOFTSKILL_TRANSFER_SOURCE_CHECKPOINT", "")


def _blocks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_shared = torch.arange(16 * HIDDEN_SIZE, dtype=torch.float32).reshape(16, HIDDEN_SIZE)
    task = raw_shared + 1000.0
    transferred = raw_shared + 2000.0
    return raw_shared, task, transferred


def _bundle(condition: str) -> TransferPrefixParameters:
    raw_shared, task, transferred = _blocks()
    return TransferPrefixParameters(
        raw_shared,
        task,
        condition=condition,
        transferred_shared_block=(transferred if condition == CONDITION_F1 else None),
    )


def _wrapper(
    condition: str,
    *,
    task_name: str = "chartqa",
    bundle: TransferPrefixParameters | None = None,
) -> TargetOnlyTransferSoftPrefixVisionLM:
    model = object.__new__(TargetOnlyTransferSoftPrefixVisionLM)
    model.model = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    model.prefix_parameters = bundle or _bundle(condition)
    model.model_name = "Qwen/Qwen3.5-4B"
    model.task_name = task_name
    model.condition = condition
    model.prefix_length = 16
    model.use_residual_reparameterization = False
    model.initialization_audit = {"task_name": task_name}
    model.source_metadata = (
        {"state_key": SHARED_CHECKPOINT_STATE_KEY, "tensor_sha256": "fixture"}
        if condition == CONDITION_F1
        else {}
    )
    model.validate_invariants()
    return model


def _source_checkpoint(shared: torch.Tensor, **overrides) -> dict:
    model_state = {
        "protocol_version": "v2",
        "layout": "shared16_task16",
        "initialization_mode": "behavior_markdown",
        "prefix_length": 16,
        "use_residual_reparameterization": False,
        "shared_prefix_embeddings": shared,
        "task_prefix_embeddings": {
            "searchqa": torch.full((16, shared.shape[-1]), 999.0, dtype=shared.dtype)
        },
    }
    model_state.update(overrides)
    return {
        "model": model_state,
        "optimizer": {"state": {"task_only_marker": torch.tensor(123.0)}},
        "history": [{"should_not_be_loaded": True}],
    }


class _AuditTokenizer:
    def __init__(self) -> None:
        shared = json.loads(SHARED_TOKEN_AUDIT_PATH.read_text(encoding="utf-8"))
        tasks = json.loads(TASK_TOKEN_AUDIT_PATH.read_text(encoding="utf-8"))
        shared_record = shared["files"]["shared_behavior.md"]
        self.by_text = {
            shared_record["raw_text"]: shared_record["selected_token_ids"]
        }
        for filename in ("chartqa_behavior.md", "drop_behavior.md"):
            record = tasks["files"][filename]
            self.by_text[record["raw_text"]] = record["selected_token_ids"]

    def __call__(self, text, *, add_special_tokens, return_tensors):
        if add_special_tokens is not False or return_tensors != "pt":
            raise AssertionError("unexpected tokenizer options")
        return {"input_ids": torch.tensor([self.by_text[text]], dtype=torch.long)}

    def decode(self, token_ids):
        return " ".join(str(value) for value in token_ids)


class _Embedding(torch.nn.Module):
    embedding_dim = HIDDEN_SIZE

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, HIDDEN_SIZE), requires_grad=False)

    def forward(self, input_ids):
        return input_ids.to(dtype=self.weight.dtype).unsqueeze(-1).repeat(1, 1, HIDDEN_SIZE)


class PrefixStructureTests(unittest.TestCase):
    def test_all_conditions_are_32_tokens_with_frozen_base_and_no_residual(self) -> None:
        for condition in (CONDITION_F0, CONDITION_F1, CONDITION_I32):
            with self.subTest(condition=condition):
                model = _wrapper(condition)
                self.assertEqual(
                    tuple(model.active_prefix_embeddings().shape),
                    (EFFECTIVE_PREFIX_LENGTH, HIDDEN_SIZE),
                )
                self.assertTrue(all(not p.requires_grad for p in model.model.parameters()))
                self.assertIs(model.use_residual_reparameterization, False)

    def test_f0_only_task16_is_trainable(self) -> None:
        bundle = _bundle(CONDITION_F0)
        self.assertEqual(bundle.trainable_parameter_names(), ["task_prefix"])
        self.assertFalse(bundle.shared_prefix_tensor().requires_grad)
        self.assertTrue(bundle.task_prefix_tensor().requires_grad)

    def test_f1_uses_transferred_shared_and_only_task16_is_trainable(self) -> None:
        raw_shared, task, transferred = _blocks()
        bundle = TransferPrefixParameters(
            raw_shared,
            task,
            condition=CONDITION_F1,
            transferred_shared_block=transferred,
        )
        self.assertTrue(torch.equal(bundle.shared_prefix_tensor(), transferred))
        self.assertFalse(torch.equal(bundle.shared_prefix_tensor(), raw_shared))
        self.assertTrue(torch.equal(bundle.task_prefix_tensor(), task))
        self.assertEqual(bundle.trainable_parameter_names(), ["task_prefix"])

    def test_i32_all_32_vectors_are_one_trainable_target_parameter(self) -> None:
        bundle = _bundle(CONDITION_I32)
        self.assertEqual(bundle.trainable_parameter_names(), ["independent_prefix"])
        self.assertEqual(tuple(bundle.independent_prefix.shape), (32, HIDDEN_SIZE))
        self.assertTrue(bundle.shared_prefix_tensor().requires_grad)
        self.assertTrue(bundle.task_prefix_tensor().requires_grad)

    def test_f0_and_i32_start_numerically_identical(self) -> None:
        f0 = _bundle(CONDITION_F0)
        i32 = _bundle(CONDITION_I32)
        self.assertTrue(
            torch.equal(f0.active_prefix_embeddings(), i32.active_prefix_embeddings())
        )
        self.assertFalse(f0.shared_prefix_tensor().requires_grad)
        self.assertTrue(i32.shared_prefix_tensor().requires_grad)

    def test_only_f1_accepts_transferred_shared(self) -> None:
        raw_shared, task, transferred = _blocks()
        with self.assertRaises(ValueError):
            TransferPrefixParameters(raw_shared, task, condition=CONDITION_F1)
        for condition in (CONDITION_F0, CONDITION_I32):
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                TransferPrefixParameters(
                    raw_shared,
                    task,
                    condition=condition,
                    transferred_shared_block=transferred,
                )

    def test_chartqa_and_drop_embed_their_own_behavior_files(self) -> None:
        tokenizer = _AuditTokenizer()
        embedding = _Embedding()
        chart_shared, chart_task, chart_audit = embed_task_behavior_blocks(
            tokenizer=tokenizer,
            embedding_layer=embedding,
            device=torch.device("cpu"),
            task_name="chartqa",
        )
        drop_shared, drop_task, drop_audit = embed_task_behavior_blocks(
            tokenizer=tokenizer,
            embedding_layer=embedding,
            device=torch.device("cpu"),
            task_name="drop",
        )
        self.assertTrue(torch.equal(chart_shared, drop_shared))
        self.assertFalse(torch.equal(chart_task, drop_task))
        self.assertEqual(Path(chart_audit["task_behavior"]["path"]), TASK_BEHAVIOR_PATHS["chartqa"])
        self.assertEqual(Path(drop_audit["task_behavior"]["path"]), TASK_BEHAVIOR_PATHS["drop"])
        self.assertEqual(Path(chart_audit["shared_behavior"]["path"]), SHARED_BEHAVIOR_PATH)


class SharedCheckpointTests(unittest.TestCase):
    def _write(self, checkpoint: dict, root: str, name: str = "source.pt") -> Path:
        path = Path(root) / name
        torch.save(checkpoint, path)
        return path

    def test_extracts_only_actual_shared_key_and_records_metadata(self) -> None:
        shared = torch.arange(16 * HIDDEN_SIZE, dtype=torch.float32).reshape(
            1, 16, HIDDEN_SIZE
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self._write(_source_checkpoint(shared), tmp_dir)
            extracted, metadata = extract_shared_prefix(
                path,
                expected_hidden_size=HIDDEN_SIZE,
                expected_dtype=torch.float32,
                torch_module=torch,
            )
        self.assertTrue(torch.equal(extracted, shared[0]))
        self.assertEqual(metadata["state_key"], "model.shared_prefix_embeddings")
        self.assertEqual(metadata["stored_shape"], [1, 16, HIDDEN_SIZE])
        self.assertEqual(metadata["extracted_shape"], [16, HIDDEN_SIZE])
        self.assertEqual(metadata["dtype"], "float32")

    def test_missing_shared_key_never_falls_back_to_task_tensor(self) -> None:
        shared = torch.zeros(1, 16, HIDDEN_SIZE)
        checkpoint = _source_checkpoint(shared)
        del checkpoint["model"]["shared_prefix_embeddings"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self._write(checkpoint, tmp_dir)
            with self.assertRaises(KeyError):
                extract_shared_prefix(
                    path,
                    expected_hidden_size=HIDDEN_SIZE,
                    expected_dtype=torch.float32,
                    torch_module=torch,
                )

    def test_wrong_shared_shape_fails_fast(self) -> None:
        checkpoint = _source_checkpoint(torch.zeros(1, 15, HIDDEN_SIZE))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self._write(checkpoint, tmp_dir)
            with self.assertRaises(ValueError):
                extract_shared_prefix(
                    path,
                    expected_hidden_size=HIDDEN_SIZE,
                    expected_dtype=torch.float32,
                    torch_module=torch,
                )

    def test_wrong_shared_dtype_fails_fast(self) -> None:
        checkpoint = _source_checkpoint(torch.zeros(1, 16, HIDDEN_SIZE, dtype=torch.float64))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self._write(checkpoint, tmp_dir)
            with self.assertRaises(TypeError):
                extract_shared_prefix(
                    path,
                    expected_hidden_size=HIDDEN_SIZE,
                    expected_dtype=torch.float32,
                    torch_module=torch,
                )

    def test_candidates_are_documentation_not_auto_selection(self) -> None:
        self.assertEqual(len(SOURCE_CHECKPOINT_CANDIDATES), 3)
        self.assertTrue(all(path.endswith("best_v2.pt") for path in SOURCE_CHECKPOINT_CANDIDATES))
        self.assertEqual(SHARED_CHECKPOINT_STATE_KEY, "model.shared_prefix_embeddings")


@unittest.skipUnless(
    REAL_SHARED_CHECKPOINT,
    "SOFTSKILL_TRANSFER_SOURCE_CHECKPOINT is not set",
)
class RealSharedCheckpointCompatibilityTest(unittest.TestCase):
    def test_real_v2_shared_checkpoint_is_strictly_compatible(self) -> None:
        shared, metadata = extract_shared_prefix(
            REAL_SHARED_CHECKPOINT,
            expected_hidden_size=2560,
            expected_dtype=torch.bfloat16,
            torch_module=torch,
        )
        self.assertEqual(tuple(shared.shape), (16, 2560))
        self.assertEqual(shared.dtype, torch.bfloat16)
        self.assertEqual(metadata["state_key"], "model.shared_prefix_embeddings")
        self.assertEqual(metadata["stored_shape"], [1, 16, 2560])


class PrefixStateRoundTripTests(unittest.TestCase):
    def test_state_roundtrip_preserves_condition_freezing_and_tensors(self) -> None:
        for condition in (CONDITION_F0, CONDITION_F1, CONDITION_I32):
            with self.subTest(condition=condition):
                source = _wrapper(condition)
                buffer = io.BytesIO()
                torch.save({"model": source.state_dict()}, buffer)
                buffer.seek(0)
                loaded = torch.load(buffer, map_location="cpu", weights_only=True)["model"]

                raw_shared, task, transferred = _blocks()
                raw_shared.zero_()
                task.zero_()
                transferred.zero_()
                target_bundle = TransferPrefixParameters(
                    raw_shared,
                    task,
                    condition=condition,
                    transferred_shared_block=(
                        transferred if condition == CONDITION_F1 else None
                    ),
                )
                target = _wrapper(condition, bundle=target_bundle)
                target.load_state_dict(loaded)

                self.assertEqual(target.condition, condition)
                self.assertTrue(
                    torch.equal(
                        target.active_prefix_embeddings(),
                        source.active_prefix_embeddings(),
                    )
                )
                self.assertEqual(
                    target.trainable_parameter_names(),
                    source.trainable_parameter_names(),
                )
                if condition != CONDITION_I32:
                    self.assertFalse(target.shared_prefix_tensor().requires_grad)

    def test_restore_rejects_different_condition(self) -> None:
        state = _wrapper(CONDITION_F0).state_dict()
        with self.assertRaises(ValueError):
            _wrapper(CONDITION_F1).load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
