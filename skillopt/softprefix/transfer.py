"""Utilities for transferring soft prefixes across model embedding spaces."""
from __future__ import annotations

from typing import Any


def project_prefix_via_vocab(
    source_prefix: Any,
    source_embeddings: Any,
    target_embeddings: Any,
    *,
    temperature: float = 0.05,
    top_k: int = 0,
    normalize: bool = True,
) -> Any:
    """Project source soft-prefix rows into a target model's hidden space.

    The source prefix is first converted into a distribution over source-model
    vocabulary rows, then that distribution is used to mix target-model
    vocabulary embeddings. This requires matching token ids across both
    embedding tables.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "Soft-prefix transfer requires `torch`. Install the soft-prefix "
            "extras with `pip install -e '.[softprefix]'`."
        ) from exc

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    source_prefix = torch.as_tensor(source_prefix, dtype=torch.float32)
    source_prefix_shape = tuple(source_prefix.shape)
    if source_prefix.ndim == 3:
        source_prefix = source_prefix.flatten(0, 1)
    source_embeddings = torch.as_tensor(source_embeddings, dtype=torch.float32)
    target_dtype = getattr(target_embeddings, "dtype", torch.float32)
    target_embeddings = torch.as_tensor(target_embeddings, dtype=torch.float32)

    if source_prefix.ndim != 2:
        raise ValueError(f"source_prefix must be rank 2, got shape={tuple(source_prefix.shape)}")
    if source_embeddings.ndim != 2:
        raise ValueError(f"source_embeddings must be rank 2, got shape={tuple(source_embeddings.shape)}")
    if target_embeddings.ndim != 2:
        raise ValueError(f"target_embeddings must be rank 2, got shape={tuple(target_embeddings.shape)}")
    if int(source_prefix.shape[1]) != int(source_embeddings.shape[1]):
        raise ValueError(
            "source_prefix hidden dim "
            f"{int(source_prefix.shape[1])} != source embedding dim {int(source_embeddings.shape[1])}"
        )
    if int(source_embeddings.shape[0]) != int(target_embeddings.shape[0]):
        raise ValueError(
            "source and target vocab sizes must match: "
            f"{int(source_embeddings.shape[0])} != {int(target_embeddings.shape[0])}"
        )

    if normalize:
        query = F.normalize(source_prefix, dim=-1)
        keys = F.normalize(source_embeddings, dim=-1)
    else:
        query = source_prefix
        keys = source_embeddings

    logits = query @ keys.T
    logits = logits / float(temperature)

    if top_k:
        k = min(int(top_k), int(logits.shape[-1]))
        keep_values, keep_indices = torch.topk(logits, k=k, dim=-1)
        sparse_logits = torch.full_like(logits, float("-inf"))
        logits = sparse_logits.scatter(dim=-1, index=keep_indices, src=keep_values)

    weights = torch.softmax(logits, dim=-1)
    projected = weights @ target_embeddings
    projected = projected.to(dtype=target_dtype)
    if len(source_prefix_shape) == 3:
        projected = projected.reshape(source_prefix_shape[0], source_prefix_shape[1], -1)
    return projected
