from __future__ import annotations


def test_markdown_blocks_are_consecutive_and_repeat_only_when_needed() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import split_markdown_token_embeddings

    embeddings = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    blocks = split_markdown_token_embeddings(embeddings, prefix_length=2, num_blocks=2)

    assert blocks.shape == (2, 2, 2)
    assert torch.equal(blocks[0], embeddings[:2])
    assert torch.equal(blocks[1], embeddings[2:4])


def test_shared_initialization_is_order_invariant_and_norm_calibrated() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import positionwise_shared_initialization

    blocks = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 4.0]],
            [[0.0, 2.0], [4.0, 0.0]],
            [[2.0, 2.0], [0.0, -4.0]],
        ]
    )
    shared = positionwise_shared_initialization(blocks)
    permuted = positionwise_shared_initialization(blocks[[2, 0, 1]])
    target_norms = blocks.norm(dim=-1).mean(dim=0)

    assert torch.allclose(shared, permuted)
    assert torch.allclose(shared.norm(dim=-1), target_norms, atol=1e-6)

    cancelling = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])
    cancelled = positionwise_shared_initialization(cancelling)
    reversed_cancelled = positionwise_shared_initialization(cancelling.flip(0))
    assert torch.equal(cancelled, reversed_cancelled)
    assert torch.equal(cancelled, torch.zeros_like(cancelled))


def test_behavior_markdown_composition_uses_identical_information_in_both_layouts() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import (
        LAYOUT_INDEPENDENT,
        LAYOUT_SHARED_TASK,
        compose_behavior_markdown_blocks,
    )

    shared = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    tasks = {
        "searchqa": torch.full((2, 4), 10.0),
        "livemath": torch.full((2, 4), 20.0),
        "docvqa": torch.full((2, 4), 30.0),
    }
    no_shared, independent = compose_behavior_markdown_blocks(
        shared,
        tasks,
        layout=LAYOUT_INDEPENDENT,
    )
    stored_shared, private = compose_behavior_markdown_blocks(
        shared,
        tasks,
        layout=LAYOUT_SHARED_TASK,
    )

    assert no_shared is None
    assert torch.equal(stored_shared[0], shared)
    for task_name, task_block in tasks.items():
        assert torch.equal(independent[task_name][0], shared)
        assert torch.equal(independent[task_name][1], task_block)
        assert torch.equal(private[task_name], task_block)


def test_hybrid_initialization_uses_shared_behavior_and_hard_markdown_block2() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import (
        LAYOUT_SHARED_TASK,
        compose_hybrid_shared_behavior_task_hard_blocks,
    )

    shared = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    hard_blocks = {
        "searchqa": torch.stack([
            torch.full((2, 4), 1.0),
            torch.full((2, 4), 11.0),
        ]),
        "livemath": torch.stack([
            torch.full((2, 4), 2.0),
            torch.full((2, 4), 22.0),
        ]),
        "docvqa": torch.stack([
            torch.full((2, 4), 3.0),
            torch.full((2, 4), 33.0),
        ]),
    }

    stored_shared, private = compose_hybrid_shared_behavior_task_hard_blocks(
        shared,
        hard_blocks,
        layout=LAYOUT_SHARED_TASK,
    )

    assert torch.equal(stored_shared[0], shared)
    assert torch.equal(private["searchqa"], hard_blocks["searchqa"][1])
    assert torch.equal(private["livemath"], hard_blocks["livemath"][1])
    assert torch.equal(private["docvqa"], hard_blocks["docvqa"][1])


def test_behavior_checkpoint_embeds_markdown_needed_to_rebuild_initialization() -> None:
    from skillopt.softprefix.multitask_v2 import (
        INIT_BEHAVIOR_MARKDOWN,
        initialization_kwargs_from_checkpoint_state,
    )

    state = {
        "initialization_mode": INIT_BEHAVIOR_MARKDOWN,
        "initialization_spec": {
            "task_init_texts": {"a": "legacy a", "b": "legacy b"},
            "shared_behavior_text": "shared behavior",
            "task_behavior_texts": {"a": "private a", "b": "private b"},
        },
    }
    kwargs = initialization_kwargs_from_checkpoint_state(
        state,
        fallback_task_init_texts={"a": "changed a", "b": "changed b"},
    )

    assert kwargs == {
        "task_init_texts": {"a": "legacy a", "b": "legacy b"},
        "initialization_mode": INIT_BEHAVIOR_MARKDOWN,
        "shared_behavior_text": "shared behavior",
        "task_behavior_texts": {"a": "private a", "b": "private b"},
    }


def test_hybrid_checkpoint_embeds_hard_and_shared_markdown_sources() -> None:
    from skillopt.softprefix.multitask_v2 import (
        INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD,
        initialization_kwargs_from_checkpoint_state,
    )

    state = {
        "initialization_mode": INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD,
        "initialization_spec": {
            "task_init_texts": {"a": "hard a", "b": "hard b"},
            "shared_behavior_text": "shared behavior",
            "task_behavior_texts": {},
        },
    }
    kwargs = initialization_kwargs_from_checkpoint_state(
        state,
        fallback_task_init_texts={"a": "changed a", "b": "changed b"},
    )

    assert kwargs == {
        "task_init_texts": {"a": "hard a", "b": "hard b"},
        "initialization_mode": INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD,
        "shared_behavior_text": "shared behavior",
    }


def test_markdown_token_audit_reports_repetition_and_truncation() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import JointV2SoftPrefixVisionLM

    class FakeTokenizer:
        def __call__(self, text, **_kwargs):
            values = [int(value) for value in text.split()]
            return {"input_ids": torch.tensor([values], dtype=torch.long)}

        def decode(self, values):
            return " ".join(str(value) for value in values)

    model = JointV2SoftPrefixVisionLM.__new__(JointV2SoftPrefixVisionLM)
    model.tokenizer = FakeTokenizer()

    repeated = model._markdown_token_audit("1 2", selected_length=5)
    truncated = model._markdown_token_audit("1 2 3 4", selected_length=3)

    assert repeated == {
        "raw_text": "1 2",
        "source_token_ids": [1, 2],
        "source_token_count": 2,
        "selected_token_count": 5,
        "repeated": True,
        "truncated": False,
        "selected_token_ids": [1, 2, 1, 2, 1],
        "selected_text": "1 2 1 2 1",
    }
    assert truncated["selected_token_ids"] == [1, 2, 3]
    assert truncated["repeated"] is False
    assert truncated["truncated"] is True

    model.prefix_length = 2
    block2 = model._markdown_block_audit("1 2 3 4 5 6", block_index=1)
    assert block2["selected_token_offset"] == 2
    assert block2["selected_token_ids"] == [3, 4]
    assert block2["selected_text"] == "3 4"


def test_v2_accumulation_restores_effective_batches_and_macro_weights() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import (
        effective_optimizer_steps,
        supervised_token_count,
        token_weighted_microbatch_scales,
    )

    # SearchQA: 50 batches / accum 1; LiveMath: 9 / accum 2;
    # DocVQA: 107 / accum 4.  SearchQA determines the joint epoch length.
    assert effective_optimizer_steps(50, 1) == 50
    assert effective_optimizer_steps(9, 2) == 5
    assert effective_optimizer_steps(107, 4) == 27

    full = token_weighted_microbatch_scales([4, 4], num_tasks=3)
    partial = token_weighted_microbatch_scales([4, 3], num_tasks=3)
    assert sum(full) == 1 / 3
    assert sum(partial) == 1 / 3
    assert partial == [4 / 21, 3 / 21]
    assert partial[0] / 4 == partial[1] / 3

    # Position zero has no preceding logit and the final right-padding label is
    # ignored, exactly matching _masked_causal_lm_loss.
    labels = torch.tensor([[7, -100, 8, 9], [-100, 3, -100, -100]])
    assert supervised_token_count(torch, {"labels": labels}) == 3


def test_v2_independent_and_shared_layouts_both_expose_32_tokens() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import (
        JointV2SoftPrefixVisionLM,
        LAYOUT_INDEPENDENT,
        LAYOUT_SHARED_TASK,
    )

    independent = JointV2SoftPrefixVisionLM.__new__(JointV2SoftPrefixVisionLM)
    independent.torch = torch
    independent.layout = LAYOUT_INDEPENDENT
    independent.use_residual_reparameterization = False
    independent.task_prefix_embeddings = {
        "searchqa": torch.nn.Parameter(torch.zeros(2, 16, 8)),
    }
    independent.active_task = "searchqa"

    shared_task = JointV2SoftPrefixVisionLM.__new__(JointV2SoftPrefixVisionLM)
    shared_task.torch = torch
    shared_task.layout = LAYOUT_SHARED_TASK
    shared_task.use_residual_reparameterization = False
    shared_task.prefix_embeddings = torch.nn.Parameter(torch.zeros(1, 16, 8))
    shared_task.task_prefix_embeddings = {
        "searchqa": torch.nn.Parameter(torch.zeros(16, 8)),
    }
    shared_task.active_task = "searchqa"

    assert independent.active_prefix_embeddings().shape == (32, 8)
    assert shared_task.active_prefix_embeddings().shape == (32, 8)


def test_behavior_shared_prefix_is_identical_for_all_three_tasks() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import JointV2SoftPrefixVisionLM, LAYOUT_SHARED_TASK

    model = JointV2SoftPrefixVisionLM.__new__(JointV2SoftPrefixVisionLM)
    model.torch = torch
    model.layout = LAYOUT_SHARED_TASK
    model.use_residual_reparameterization = False
    model.prefix_embeddings = torch.nn.Parameter(torch.randn(1, 16, 8))
    model.task_prefix_embeddings = torch.nn.ParameterDict({
        "searchqa": torch.nn.Parameter(torch.randn(16, 8)),
        "livemath": torch.nn.Parameter(torch.randn(16, 8)),
        "docvqa": torch.nn.Parameter(torch.randn(16, 8)),
    })
    model.active_task = "searchqa"

    prefixes = {}
    for name in model.task_prefix_embeddings:
        model.set_active_task(name)
        prefixes[name] = model.active_raw_prefix_embeddings().detach().clone()

    for prefix in prefixes.values():
        assert prefix.shape == (32, 8)
        assert torch.equal(prefix[:16], model.prefix_embeddings[0])
    assert not torch.equal(prefixes["searchqa"][16:], prefixes["livemath"][16:])


def test_one_matched_residual_mlp_is_applied_to_the_full_prompt() -> None:
    import torch

    from skillopt.softprefix.model import ResidualPromptMLP
    from skillopt.softprefix.multitask_v2 import JointV2SoftPrefixVisionLM, LAYOUT_SHARED_TASK

    model = JointV2SoftPrefixVisionLM.__new__(JointV2SoftPrefixVisionLM)
    model.torch = torch
    model.layout = LAYOUT_SHARED_TASK
    model.use_residual_reparameterization = True
    model.residual_mode = "global_matched"
    model.prefix_embeddings = torch.nn.Parameter(torch.randn(1, 16, 8))
    model.task_prefix_embeddings = {
        "searchqa": torch.nn.Parameter(torch.randn(16, 8)),
    }
    model.active_task = "searchqa"
    model.residual_mlp = ResidualPromptMLP.build(torch, embedding_dim=8, bottleneck_size=4)

    raw = torch.cat([model.prefix_embeddings.flatten(0, 1), model.task_prefix_embeddings["searchqa"]])
    actual = model.active_prefix_embeddings()

    # Zero-initialized up projection makes the residual map an exact identity.
    assert actual.shape == (32, 8)
    assert torch.allclose(actual, raw)
    assert len(model.reparameterization_parameters()) == 6


def test_residual_mlp_initialization_uses_an_isolated_deterministic_seed() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import build_residual_mlp_deterministic

    torch.manual_seed(123)
    torch.nn.init.normal_(torch.empty(2, 16, 8))
    first = build_residual_mlp_deterministic(
        torch,
        embedding_dim=8,
        bottleneck_size=4,
        seed=99,
    )
    torch.manual_seed(123)
    torch.nn.init.normal_(torch.empty(1, 16, 8))
    second = build_residual_mlp_deterministic(
        torch,
        embedding_dim=8,
        bottleneck_size=4,
        seed=99,
    )

    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])

    torch.manual_seed(777)
    expected_next_draw = torch.rand(5)
    torch.manual_seed(777)
    build_residual_mlp_deterministic(
        torch,
        embedding_dim=8,
        bottleneck_size=4,
        seed=99,
    )
    assert torch.equal(expected_next_draw, torch.rand(5))


def test_branch_checkpoint_roundtrip_restores_all_trainable_state() -> None:
    import torch

    from skillopt.softprefix.model import ResidualPromptMLP
    from skillopt.softprefix.multitask_v2 import (
        PROTOCOL_VERSION,
        JointV2SoftPrefixVisionLM,
        LAYOUT_SHARED_TASK,
        RESIDUAL_BRANCH_DECOUPLED,
    )

    def make_model(fill: float) -> JointV2SoftPrefixVisionLM:
        model = JointV2SoftPrefixVisionLM.__new__(JointV2SoftPrefixVisionLM)
        model.torch = torch
        model.protocol_version = PROTOCOL_VERSION
        model.layout = LAYOUT_SHARED_TASK
        model.prefix_length = 16
        model.residual_bottleneck_size = 4
        model.use_residual_reparameterization = True
        model.residual_mode = RESIDUAL_BRANCH_DECOUPLED
        model.residual_init_seed = 1234
        model.device = torch.device("cpu")
        model.prefix_embeddings = torch.nn.Parameter(torch.full((1, 16, 8), fill))
        model.task_prefix_embeddings = torch.nn.ParameterDict({
            "searchqa": torch.nn.Parameter(torch.full((16, 8), fill + 1)),
            "livemath": torch.nn.Parameter(torch.full((16, 8), fill + 2)),
        })
        model.shared_residual_mlp = ResidualPromptMLP.build(torch, 8, 4)
        model.task_residual_mlps = torch.nn.ModuleDict({
            name: ResidualPromptMLP.build(torch, 8, 4)
            for name in model.task_prefix_embeddings
        })
        model.active_task = "livemath"
        return model

    source = make_model(2.0)
    target = make_model(-3.0)
    with torch.no_grad():
        source.shared_residual_mlp[0].weight.fill_(0.125)
        source.task_residual_mlps["livemath"][0].bias.fill_(0.25)

    saved = source.state_dict()
    target.load_state_dict(saved)

    assert torch.equal(target.prefix_embeddings, source.prefix_embeddings)
    for name in source.task_prefix_embeddings:
        assert torch.equal(
            target.task_prefix_embeddings[name],
            source.task_prefix_embeddings[name],
        )
    for expected, actual in zip(
        source.reparameterization_parameters(),
        target.reparameterization_parameters(),
    ):
        assert torch.equal(actual, expected)
    assert target.active_task == "livemath"


def test_progressive_branch_mode_freezes_the_complete_task_path_at_start() -> None:
    import torch

    from skillopt.softprefix.model import ResidualPromptMLP
    from skillopt.softprefix.multitask_v2 import (
        JointV2SoftPrefixVisionLM,
        LAYOUT_SHARED_TASK,
        RESIDUAL_BRANCH_DECOUPLED,
        _build_optimizer,
    )

    model = JointV2SoftPrefixVisionLM.__new__(JointV2SoftPrefixVisionLM)
    model.torch = torch
    model.layout = LAYOUT_SHARED_TASK
    model.use_residual_reparameterization = True
    model.residual_mode = RESIDUAL_BRANCH_DECOUPLED
    model.prefix_embeddings = torch.nn.Parameter(torch.randn(1, 16, 8))
    model.task_prefix_embeddings = torch.nn.ParameterDict({
        "searchqa": torch.nn.Parameter(torch.randn(16, 8)),
    })
    model.active_task = "searchqa"
    model.shared_residual_mlp = ResidualPromptMLP.build(torch, 8, 4)
    model.task_residual_mlps = torch.nn.ModuleDict({
        "searchqa": ResidualPromptMLP.build(torch, 8, 4),
    })
    optimizer = _build_optimizer(
        torch,
        model,
        learning_rate=1e-3,
        lr_schedule="progressive",
        shared_lr_start=1e-3,
    )
    task_before = model.task_prefix_embeddings["searchqa"].detach().clone()
    task_mlp_before = {
        name: value.detach().clone()
        for name, value in model.task_residual_mlps["searchqa"].state_dict().items()
    }

    model.active_prefix_embeddings().square().mean().backward()
    optimizer.step()

    assert torch.equal(task_before, model.task_prefix_embeddings["searchqa"].detach())
    for name, value in model.task_residual_mlps["searchqa"].state_dict().items():
            assert torch.equal(task_mlp_before[name], value)


def test_behavior_two_stage_freezes_tasks_then_isolates_private_gradients() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import (
        BEHAVIOR_STAGE_SHARED_TASK,
        BEHAVIOR_STAGE_SHARED_WARMUP,
        INIT_BEHAVIOR_MARKDOWN,
        LR_BEHAVIOR_TWO_STAGE,
        JointV2SoftPrefixVisionLM,
        LAYOUT_SHARED_TASK,
        _build_optimizer,
        behavior_stage_for_step,
        behavior_warmup_steps,
        set_behavior_training_stage,
    )

    model = JointV2SoftPrefixVisionLM.__new__(JointV2SoftPrefixVisionLM)
    model.torch = torch
    model.layout = LAYOUT_SHARED_TASK
    model.initialization_mode = INIT_BEHAVIOR_MARKDOWN
    model.use_residual_reparameterization = False
    model.prefix_embeddings = torch.nn.Parameter(torch.randn(1, 16, 8))
    model.task_prefix_embeddings = torch.nn.ParameterDict({
        "searchqa": torch.nn.Parameter(torch.randn(16, 8)),
        "livemath": torch.nn.Parameter(torch.randn(16, 8)),
        "docvqa": torch.nn.Parameter(torch.randn(16, 8)),
    })
    model.active_task = "searchqa"
    optimizer = _build_optimizer(
        torch,
        model,
        learning_rate=1e-3,
        lr_schedule=LR_BEHAVIOR_TWO_STAGE,
        shared_lr_start=1e-3,
    )

    assert behavior_warmup_steps(150, 0.2) == 30
    assert behavior_stage_for_step(29, total_steps=150) == BEHAVIOR_STAGE_SHARED_WARMUP
    assert behavior_stage_for_step(30, total_steps=150) == BEHAVIOR_STAGE_SHARED_TASK

    task_before = {
        name: parameter.detach().clone()
        for name, parameter in model.task_prefix_embeddings.items()
    }
    warmup = set_behavior_training_stage(
        model,
        optimizer,
        stage=BEHAVIOR_STAGE_SHARED_WARMUP,
    )
    assert warmup["optimizer_lrs"] == {"shared": 1e-3, "task": 0.0}
    assert not any(warmup["task_trainable"].values())
    model.active_raw_prefix_embeddings().square().mean().backward()
    optimizer.step()
    for name, parameter in model.task_prefix_embeddings.items():
        assert torch.equal(task_before[name], parameter.detach())

    optimizer.zero_grad(set_to_none=True)
    joint = set_behavior_training_stage(
        model,
        optimizer,
        stage=BEHAVIOR_STAGE_SHARED_TASK,
    )
    assert joint["optimizer_lrs"] == {"shared": 1e-4, "task": 1e-3}
    assert all(joint["task_trainable"].values())
    model.set_active_task("livemath")
    model.active_raw_prefix_embeddings().square().mean().backward()
    assert model.prefix_embeddings.grad is not None
    assert model.task_prefix_embeddings["livemath"].grad is not None
    assert model.task_prefix_embeddings["searchqa"].grad is None
    assert model.task_prefix_embeddings["docvqa"].grad is None


def test_behavior_two_stage_global_residual_is_frozen_then_jointly_trained() -> None:
    import torch

    from skillopt.softprefix.model import ResidualPromptMLP
    from skillopt.softprefix.multitask_v2 import (
        BEHAVIOR_STAGE_SHARED_TASK,
        BEHAVIOR_STAGE_SHARED_WARMUP,
        INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD,
        LR_BEHAVIOR_TWO_STAGE,
        RESIDUAL_GLOBAL_MATCHED,
        JointV2SoftPrefixVisionLM,
        LAYOUT_SHARED_TASK,
        _build_optimizer,
        set_behavior_training_stage,
    )

    model = JointV2SoftPrefixVisionLM.__new__(JointV2SoftPrefixVisionLM)
    model.torch = torch
    model.layout = LAYOUT_SHARED_TASK
    model.initialization_mode = INIT_HYBRID_SHARED_BEHAVIOR_TASK_HARD
    model.use_residual_reparameterization = True
    model.residual_mode = RESIDUAL_GLOBAL_MATCHED
    model.prefix_embeddings = torch.nn.Parameter(torch.randn(1, 16, 8))
    model.task_prefix_embeddings = torch.nn.ParameterDict({
        "searchqa": torch.nn.Parameter(torch.randn(16, 8)),
        "livemath": torch.nn.Parameter(torch.randn(16, 8)),
        "docvqa": torch.nn.Parameter(torch.randn(16, 8)),
    })
    model.active_task = "searchqa"
    model.residual_mlp = ResidualPromptMLP.build(
        torch,
        embedding_dim=8,
        bottleneck_size=4,
    )
    optimizer = _build_optimizer(
        torch,
        model,
        learning_rate=1e-3,
        lr_schedule=LR_BEHAVIOR_TWO_STAGE,
        shared_lr_start=1e-3,
    )

    task_before = {
        name: parameter.detach().clone()
        for name, parameter in model.task_prefix_embeddings.items()
    }
    residual_before = {
        name: value.detach().clone()
        for name, value in model.residual_mlp.state_dict().items()
    }
    warmup = set_behavior_training_stage(
        model,
        optimizer,
        stage=BEHAVIOR_STAGE_SHARED_WARMUP,
        residual_joint_lr=1e-3,
    )
    assert warmup["optimizer_lrs"] == {
        "shared": 1e-3,
        "task": 0.0,
        "residual": 0.0,
    }
    assert not warmup["residual_trainable"]["all_parameters"]
    model.active_prefix_embeddings().square().mean().backward()
    optimizer.step()
    for name, parameter in model.task_prefix_embeddings.items():
        assert torch.equal(task_before[name], parameter.detach())
    for name, value in model.residual_mlp.state_dict().items():
        assert torch.equal(residual_before[name], value)

    optimizer.zero_grad(set_to_none=True)
    joint = set_behavior_training_stage(
        model,
        optimizer,
        stage=BEHAVIOR_STAGE_SHARED_TASK,
        residual_joint_lr=1e-3,
    )
    assert joint["optimizer_lrs"] == {
        "shared": 1e-4,
        "task": 1e-3,
        "residual": 1e-3,
    }
    assert joint["residual_trainable"]["all_parameters"]
    model.set_active_task("livemath")
    model.active_prefix_embeddings().square().mean().backward()
    assert model.prefix_embeddings.grad is not None
    assert model.task_prefix_embeddings["livemath"].grad is not None
    assert model.task_prefix_embeddings["searchqa"].grad is None
    assert model.task_prefix_embeddings["docvqa"].grad is None
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().max()) > 0.0
        for parameter in model.residual_mlp.parameters()
    )


def test_four_state_artifact_separates_training_and_residual_changes() -> None:
    import torch

    from skillopt.softprefix.multitask_v2 import (
        PROTOCOL_VERSION,
        JointV2SoftPrefixVisionLM,
        LAYOUT_SHARED_TASK,
        RESIDUAL_GLOBAL_MATCHED,
        build_prefix_state_artifact,
        capture_prefix_snapshot,
    )

    model = JointV2SoftPrefixVisionLM.__new__(JointV2SoftPrefixVisionLM)
    model.torch = torch
    model.protocol_version = PROTOCOL_VERSION
    model.layout = LAYOUT_SHARED_TASK
    model.prefix_length = 2
    model.use_residual_reparameterization = True
    model.residual_mode = RESIDUAL_GLOBAL_MATCHED
    model.residual_bottleneck_size = 3
    model.residual_init_seed = 19
    model.prefix_embeddings = torch.nn.Parameter(torch.ones(1, 2, 4))
    model.task_prefix_embeddings = torch.nn.ParameterDict({
        "a": torch.nn.Parameter(torch.full((2, 4), 2.0)),
        "b": torch.nn.Parameter(torch.full((2, 4), 3.0)),
    })
    model.active_task = "a"
    # Identity as the residual branch makes Phi(P)=2P and gives an exact,
    # easily audited non-zero residual contribution.
    model.residual_mlp = torch.nn.Identity()

    initial = capture_prefix_snapshot(model, ["a", "b"])
    with torch.no_grad():
        model.prefix_embeddings.add_(0.5)
        for parameter in model.task_prefix_embeddings.values():
            parameter.add_(1.0)
    artifact = build_prefix_state_artifact(
        model,
        ["a", "b"],
        initial,
        source_checkpoint="best_v2.pt",
    )

    states = artifact["states"]
    assert torch.equal(states["initial"]["a"][:2], torch.ones(2, 4))
    assert torch.equal(states["final_raw"]["a"][:2], torch.full((2, 4), 1.5))
    assert torch.equal(
        states["residual_contribution"]["a"],
        states["final_raw"]["a"],
    )
    assert torch.equal(
        states["folded"]["a"],
        states["final_raw"]["a"] + states["residual_contribution"]["a"],
    )
    assert artifact["invariants"]["folded_reconstruction_max_abs_error"] == 0.0
    for difference in artifact["invariants"][
        "shared_max_abs_difference_across_tasks"
    ].values():
        assert difference == 0.0


def test_v2_entrypoint_uses_versioned_local_configs() -> None:
    from pathlib import Path

    from skillopt.config import flatten_config, load_config

    project_root = Path(__file__).resolve().parents[1]
    search_path = project_root / "configs/searchqa/soft_prefix_2x16.yaml"
    live_path = project_root / "configs/livemathematicianbench/soft_prefix_2x16.yaml"
    doc_path = project_root / "configs/docvqa/soft_prefix_2x16.yaml"
    assert search_path.is_file()
    assert live_path.is_file()
    assert doc_path.is_file()
    search = flatten_config(load_config(str(search_path)))
    live = flatten_config(load_config(str(live_path)))
    doc = flatten_config(load_config(str(doc_path)))
    assert (search["batch_size"], search["accumulation"]) == (8, 1)
    assert (live["batch_size"], live["accumulation"]) == (4, 2)
    assert (doc["batch_size"], doc["accumulation"]) == (1, 4)


def test_behavior_markdown_seeds_start_with_behavior_not_metadata() -> None:
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    behavior_root = project_root / "skillopt/behavior_compression/v1"
    expected = {
        "shared_behavior.md",
        "searchqa_behavior.md",
        "livemath_behavior.md",
        "docvqa_behavior.md",
    }

    for filename in expected:
        text = (behavior_root / filename).read_text(encoding="utf-8").strip()
        assert text
        assert not text.startswith("#")
        assert "No learned rules" not in text
        assert len(text.splitlines()) == 1

    audit = json.loads((behavior_root / "tokenizer_audit.json").read_text(encoding="utf-8"))
    assert set(audit["files"]) == expected
    for filename, record in audit["files"].items():
        assert record["raw_text"] == (behavior_root / filename).read_text(
            encoding="utf-8"
        ).strip()
        assert record["source_token_count"] == 16
        assert record["selected_token_count"] == 16
        assert record["source_token_ids"] == record["selected_token_ids"]
        assert record["selected_text"] == record["raw_text"]
        assert record["repeated"] is False
        assert record["truncated"] is False


if __name__ == "__main__":
    test_functions = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for test_name, test_function in test_functions:
        test_function()
        print(f"PASS {test_name}")
    print(f"{len(test_functions)} v2 tests passed")
