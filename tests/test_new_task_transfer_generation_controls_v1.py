"""Checks that new-task generation controls reach Transformers generate()."""
from __future__ import annotations

import unittest

import torch

from skillopt.softprefix.model import SoftPrefixVisionLM


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, *_args, **_kwargs):
        return "<answer>ok</answer>"


class _Generator:
    def __init__(self) -> None:
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[12, 13]], dtype=torch.long)


class GenerationControlTests(unittest.TestCase):
    def test_text_generation_explicitly_enables_cache_and_answer_stop(self) -> None:
        wrapper = object.__new__(SoftPrefixVisionLM)
        wrapper.torch = torch
        wrapper.tokenizer = _Tokenizer()
        wrapper.device = torch.device("cpu")
        wrapper.model = _Generator()
        wrapper._with_prefix = lambda _batch: (
            torch.zeros((1, 4, 3)),
            torch.ones((1, 4), dtype=torch.long),
            None,
            {},
        )
        result = SoftPrefixVisionLM.generate_from_prompt(
            wrapper,
            "prompt",
            max_prompt_tokens=32,
            max_new_tokens=16,
            use_prefix=True,
            stop_strings=["</answer>"],
            use_cache=True,
        )
        self.assertEqual(result, "<answer>ok</answer>")
        self.assertIs(wrapper.model.kwargs["use_cache"], True)
        self.assertEqual(wrapper.model.kwargs["stop_strings"], ["</answer>"])
        self.assertIs(wrapper.model.kwargs["tokenizer"], wrapper.tokenizer)


if __name__ == "__main__":
    unittest.main()
