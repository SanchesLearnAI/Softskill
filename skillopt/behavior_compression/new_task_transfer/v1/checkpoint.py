"""Strict Shared16 extraction from the existing three-task v2 checkpoints."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


SHARED_CHECKPOINT_STATE_KEY = "model.shared_prefix_embeddings"
SOURCE_CHECKPOINT_CANDIDATES = tuple(
    "outputs/behavior_compression/v1/"
    f"shared16_task16_raw_two_stage/seed{seed}/best_v2.pt"
    for seed in (1, 2, 3)
)
PREFIX_LENGTH = 16


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    """Hash exact tensor bytes without converting bfloat16 to float32."""
    import torch

    byte_view = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _torch_load_weights_only(torch_module: Any, source: Path) -> Any:
    try:
        return torch_module.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        return torch_module.load(source, map_location="cpu")


def extract_shared_prefix(
    checkpoint_path: str | os.PathLike[str],
    *,
    expected_hidden_size: int,
    expected_dtype: Any,
    torch_module: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load only Shared16, validate compatibility, and return a detached CPU clone.

    The current v2 file stores Shared16 as ``[1, 16, hidden]`` under
    ``checkpoint['model']['shared_prefix_embeddings']``.  The singleton storage
    dimension is accepted explicitly; the extracted tensor must be exactly
    ``[16, expected_hidden_size]``.
    """
    if torch_module is None:
        import torch as torch_module

    if int(expected_hidden_size) < 1:
        raise ValueError("expected_hidden_size must be positive")
    if expected_dtype is None:
        raise ValueError("expected_dtype must be supplied explicitly")
    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"explicit Shared checkpoint does not exist: {source}")

    checkpoint = _torch_load_weights_only(torch_module, source)
    if not isinstance(checkpoint, dict):
        raise ValueError("Shared source checkpoint must be a dictionary")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint is missing the top-level model state")
    if str(model_state.get("protocol_version")) != "v2":
        raise ValueError("Shared source checkpoint protocol_version must be v2")
    if str(model_state.get("layout")) != "shared16_task16":
        raise ValueError("Shared source checkpoint layout must be shared16_task16")
    if str(model_state.get("initialization_mode")) != "behavior_markdown":
        raise ValueError("Shared source must use behavior_markdown initialization")
    if int(model_state.get("prefix_length", -1)) != PREFIX_LENGTH:
        raise ValueError("Shared source prefix_length must be 16")
    if bool(model_state.get("use_residual_reparameterization", False)):
        raise ValueError("Shared source checkpoint must have residual disabled")

    if "shared_prefix_embeddings" not in model_state:
        raise KeyError(
            "checkpoint is missing model.shared_prefix_embeddings; Task tensors "
            "are never accepted as a Shared source"
        )
    stored = model_state["shared_prefix_embeddings"]
    if not torch_module.is_tensor(stored):
        raise TypeError("model.shared_prefix_embeddings must be a tensor")
    stored_shape = tuple(int(size) for size in stored.shape)
    if stored.dim() == 3:
        if int(stored.shape[0]) != 1:
            raise ValueError(
                "stored Shared tensor may only have one explicit outer skill dimension; "
                f"got {stored_shape}"
            )
        shared = stored[0]
    else:
        shared = stored
    expected_shape = (PREFIX_LENGTH, int(expected_hidden_size))
    if shared.dim() != 2 or tuple(int(size) for size in shared.shape) != expected_shape:
        raise ValueError(
            f"extracted Shared tensor must have shape {expected_shape}, got "
            f"{tuple(int(size) for size in shared.shape)}"
        )
    if not shared.is_floating_point():
        raise TypeError("Shared tensor must have a floating-point dtype")
    if shared.dtype != expected_dtype:
        raise TypeError(
            "Shared tensor dtype does not match the frozen base model embeddings: "
            f"checkpoint={_dtype_name(shared.dtype)}, expected={_dtype_name(expected_dtype)}"
        )

    shared = shared.detach().cpu().contiguous().clone()
    metadata = {
        "checkpoint_path": str(source),
        "checkpoint_sha256": file_sha256(source),
        "state_key": SHARED_CHECKPOINT_STATE_KEY,
        "stored_shape": list(stored_shape),
        "extracted_shape": list(shared.shape),
        "dtype": _dtype_name(shared.dtype),
        "expected_hidden_size": int(expected_hidden_size),
        "tensor_sha256": tensor_sha256(shared),
        "source_protocol_version": str(model_state["protocol_version"]),
        "source_layout": str(model_state["layout"]),
        "source_initialization_mode": str(model_state["initialization_mode"]),
        "source_residual_enabled": False,
    }
    return shared, metadata
