from __future__ import annotations


def test_progressive_learning_rates_decouple_shared_and_task() -> None:
    from skillopt.softprefix.multitask_trainer import progressive_learning_rates

    start_shared, start_task = progressive_learning_rates(
        0.0,
        shared_start=1e-3,
        shared_min=5e-5,
        shared_decay=3.0,
        task_max=1e-3,
        task_growth=5.0,
    )
    middle_shared, middle_task = progressive_learning_rates(
        0.5,
        shared_start=1e-3,
        shared_min=5e-5,
        shared_decay=3.0,
        task_max=1e-3,
        task_growth=5.0,
    )
    end_shared, end_task = progressive_learning_rates(
        1.0,
        shared_start=1e-3,
        shared_min=5e-5,
        shared_decay=3.0,
        task_max=1e-3,
        task_growth=5.0,
    )

    assert start_shared > middle_shared > end_shared
    assert start_task < middle_task < end_task
    assert start_task == 0.0
    assert end_shared >= 5e-5


def test_active_task_selects_one_task_prefix_and_shared_receives_all_gradients() -> None:
    import torch

    from skillopt.softprefix.model import SharedTaskSoftPrefixVisionLM

    model = SharedTaskSoftPrefixVisionLM.__new__(SharedTaskSoftPrefixVisionLM)
    model.torch = torch
    model.prefix_embeddings = torch.nn.Parameter(torch.tensor([[[1.0], [2.0]]]))
    model.task_prefix_embeddings = {
        "searchqa": torch.nn.Parameter(torch.tensor([[3.0], [4.0]])),
        "livemath": torch.nn.Parameter(torch.tensor([[5.0], [6.0]])),
        "docvqa": torch.nn.Parameter(torch.tensor([[7.0], [8.0]])),
    }
    model.active_task = "searchqa"

    assert model.active_prefix_embeddings().squeeze(-1).tolist() == [1.0, 2.0, 3.0, 4.0]
    model.active_prefix_embeddings().sum().backward()
    assert model.prefix_embeddings.grad is not None
    assert model.task_prefix_embeddings["searchqa"].grad is not None
    assert model.task_prefix_embeddings["livemath"].grad is None
    assert model.task_prefix_embeddings["docvqa"].grad is None

    for parameter in model.trainable_parameters():
        parameter.grad = None
    model.set_active_task("docvqa")
    model.active_prefix_embeddings().sum().backward()
    assert model.prefix_embeddings.grad is not None
    assert model.task_prefix_embeddings["docvqa"].grad is not None
    assert model.task_prefix_embeddings["searchqa"].grad is None


def test_shared16_task16_uses_a_fixed_total_prefix_budget() -> None:
    import torch

    from skillopt.softprefix.model import SharedTaskSoftPrefixVisionLM

    model = SharedTaskSoftPrefixVisionLM.__new__(SharedTaskSoftPrefixVisionLM)
    model.torch = torch
    model.prefix_length = 16
    model.prefix_embeddings = torch.nn.Parameter(torch.zeros(1, 16, 8))
    model.task_prefix_embeddings = {
        "searchqa": torch.nn.Parameter(torch.zeros(16, 8)),
    }
    model.active_task = "searchqa"

    assert model.active_prefix_embeddings().shape == (32, 8)
