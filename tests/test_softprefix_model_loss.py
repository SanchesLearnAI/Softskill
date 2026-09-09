"""Tests for soft-prefix training loss memory behavior."""
from __future__ import annotations

from types import SimpleNamespace


def test_causal_forward_computes_masked_loss_without_backbone_labels() -> None:
    import torch

    from skillopt.softprefix.model import SoftPrefixCausalLM

    class FakeBackbone(torch.nn.Module):
        dtype = torch.float32

        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.lm_head = torch.nn.Linear(4, 16, bias=False)
            self.labels_seen = "unset"

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, inputs_embeds, attention_mask, labels=None, **kwargs):
            del attention_mask, kwargs
            self.labels_seen = labels
            if labels is not None:
                raise AssertionError("wrapper should not ask HF to compute full-sequence loss")
            return SimpleNamespace(logits=self.lm_head(inputs_embeds))

    backbone = FakeBackbone()
    wrapper = SoftPrefixCausalLM.__new__(SoftPrefixCausalLM)
    wrapper.torch = torch
    wrapper.model = backbone
    wrapper.device = torch.device("cpu")
    wrapper.prefix_length = 2
    wrapper.prefix_embeddings = torch.nn.Parameter(torch.randn(2, 4))

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, 6, 7]]),
    }

    outputs = wrapper.forward(batch)

    assert backbone.labels_seen is None
    assert outputs.loss.requires_grad
    outputs.loss.backward()
    assert wrapper.prefix_embeddings.grad is not None


def test_causal_prefix_can_be_inserted_at_batch_indices() -> None:
    import torch

    from skillopt.softprefix.model import SoftPrefixCausalLM

    class FakeBackbone(torch.nn.Module):
        dtype = torch.float32

        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 1)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.arange(16, dtype=torch.float32).view(16, 1))

        def get_input_embeddings(self):
            return self.embedding

    wrapper = SoftPrefixCausalLM.__new__(SoftPrefixCausalLM)
    wrapper.torch = torch
    wrapper.model = FakeBackbone()
    wrapper.device = torch.device("cpu")
    wrapper.prefix_length = 2
    wrapper.prefix_embeddings = torch.nn.Parameter(torch.tensor([[100.0], [101.0]]))

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    labels = torch.tensor([[-100, 20, 21], [-100, -100, 22]])

    inputs_embeds, full_attention_mask, full_labels = wrapper._with_prefix(
        input_ids,
        attention_mask,
        labels,
        prefix_insert_idx=torch.tensor([1, 2]),
    )

    assert inputs_embeds.squeeze(-1).tolist() == [
        [1.0, 100.0, 101.0, 2.0, 3.0],
        [4.0, 5.0, 100.0, 101.0, 6.0],
    ]
    assert full_attention_mask.tolist() == [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
    ]
    assert full_labels.tolist() == [
        [-100, -100, -100, 20, 21],
        [-100, -100, -100, -100, 22],
    ]


def test_three_soft_skills_use_consecutive_markdown_embeddings_and_all_receive_gradients() -> None:
    import torch

    from skillopt.softprefix.model import SoftPrefixCausalLM

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            del text, kwargs
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]])}

    class FakeBackbone(torch.nn.Module):
        dtype = torch.float32

        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 2)
            self.lm_head = torch.nn.Linear(2, 16, bias=False)
            with torch.no_grad():
                values = torch.arange(32, dtype=torch.float32).reshape(16, 2)
                self.embedding.weight.copy_(values)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, inputs_embeds, attention_mask, **kwargs):
            del attention_mask, kwargs
            return SimpleNamespace(logits=self.lm_head(inputs_embeds.cumsum(dim=1)))

    wrapper = SoftPrefixCausalLM.__new__(SoftPrefixCausalLM)
    wrapper.torch = torch
    wrapper.tokenizer = FakeTokenizer()
    wrapper.model = FakeBackbone()
    wrapper.device = torch.device("cpu")
    wrapper.prefix_length = 2
    wrapper.num_soft_skills = 3
    wrapper.prefix_embeddings = torch.nn.Parameter(torch.empty(3, 2, 2))

    wrapper.initialize_from_text("markdown skill")

    expected = wrapper.model.embedding(torch.tensor([1, 2, 3, 4, 5, 6])).reshape(3, 2, 2)
    assert torch.equal(wrapper.prefix_embeddings.detach(), expected.detach())
    assert not torch.equal(wrapper.prefix_embeddings[0], wrapper.prefix_embeddings[1])

    outputs = wrapper.forward(
        {
            "input_ids": torch.tensor([[7, 8]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "labels": torch.tensor([[-100, 9]]),
        }
    )
    outputs.loss.backward()

    assert wrapper.prefix_embeddings.grad is not None
    assert wrapper.prefix_embeddings.grad.shape == (3, 2, 2)
    assert all(float(skill_grad.abs().sum()) > 0 for skill_grad in wrapper.prefix_embeddings.grad)


def test_two_by_sixteen_uses_a_fixed_total_prefix_budget() -> None:
    import torch

    from skillopt.softprefix.model import SoftPrefixCausalLM

    wrapper = SoftPrefixCausalLM.__new__(SoftPrefixCausalLM)
    wrapper.torch = torch
    wrapper.prefix_length = 16
    wrapper.num_soft_skills = 2
    wrapper.prefix_embeddings = torch.nn.Parameter(torch.zeros(2, 16, 8))

    assert wrapper.prefix_embeddings.shape == (2, 16, 8)
    assert wrapper.active_prefix_embeddings().shape == (32, 8)
