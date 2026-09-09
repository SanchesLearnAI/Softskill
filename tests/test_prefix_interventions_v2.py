from __future__ import annotations


def _assert_raises(error_type, message_fragment: str, function) -> None:
    try:
        function()
    except error_type as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message_fragment}")


def _artifact(torch):
    shared_initial = torch.full((2, 3), 1.0)
    shared_final = torch.full((2, 3), 10.0)
    task_values = {"a": 2.0, "b": 3.0, "c": 4.0}
    initial = {}
    final_raw = {}
    residual = {}
    folded = {}
    for name, task_value in task_values.items():
        initial[name] = torch.cat([shared_initial, torch.full((2, 3), task_value)], dim=0)
        final_raw[name] = torch.cat([shared_final, torch.full((2, 3), task_value * 10)], dim=0)
        residual[name] = final_raw[name] * 2
        folded[name] = final_raw[name] + residual[name]
    return {
        "protocol_version": "v2",
        "layout": "shared16_task16",
        "prefix_length": 2,
        "task_order": ["a", "b", "c"],
        "residual_mode": "global_matched",
        "states": {
            "initial": initial,
            "final_raw": final_raw,
            "residual_contribution": residual,
            "folded": folded,
        },
    }


def test_scaled_residual_endpoints_are_exact() -> None:
    import torch

    from skillopt.softprefix.interventions_v2 import scaled_residual_prompt

    raw = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    residual = torch.full_like(raw, 5.0)
    assert torch.equal(scaled_residual_prompt(raw, residual, 0.0), raw)
    assert torch.equal(scaled_residual_prompt(raw, residual, 1.0), raw + residual)


def test_restore_and_task_swap_construct_expected_raw_blocks() -> None:
    import torch

    from skillopt.softprefix.interventions_v2 import (
        RESTORE_BOTH_INITIAL,
        RESTORE_SHARED_INITIAL,
        RESTORE_TASK_INITIAL,
        construct_raw_intervention,
    )

    artifact = _artifact(torch)
    shared_initial = construct_raw_intervention(
        artifact, target_task="a", condition=RESTORE_SHARED_INITIAL
    )
    task_initial = construct_raw_intervention(
        artifact, target_task="a", condition=RESTORE_TASK_INITIAL
    )
    both_initial = construct_raw_intervention(
        artifact, target_task="a", condition=RESTORE_BOTH_INITIAL
    )
    swapped = construct_raw_intervention(artifact, target_task="a", donor_task="c")

    assert torch.equal(shared_initial[:2], torch.full((2, 3), 1.0))
    assert torch.equal(shared_initial[2:], torch.full((2, 3), 20.0))
    assert torch.equal(task_initial[:2], torch.full((2, 3), 10.0))
    assert torch.equal(task_initial[2:], torch.full((2, 3), 2.0))
    assert torch.equal(both_initial, artifact["states"]["initial"]["a"])
    assert torch.equal(swapped[:2], torch.full((2, 3), 10.0))
    assert torch.equal(swapped[2:], torch.full((2, 3), 40.0))


def test_artifact_validation_requires_identical_shared_blocks() -> None:
    import torch

    from skillopt.softprefix.interventions_v2 import validate_shared_task_artifact

    artifact = _artifact(torch)
    assert validate_shared_task_artifact(artifact, checkpoint_tasks=["a", "b", "c"]) == [
        "a",
        "b",
        "c",
    ]
    artifact["states"]["final_raw"]["b"][0, 0] += 1
    _assert_raises(
        ValueError,
        "Shared block differs",
        lambda: validate_shared_task_artifact(
            artifact, checkpoint_tasks=["a", "b", "c"]
        ),
    )


def test_final_residual_map_is_scaled_after_reparameterization() -> None:
    import torch

    from skillopt.softprefix.interventions_v2 import apply_final_residual_with_alpha

    class Dummy:
        layout = "shared16_task16"
        use_residual_reparameterization = True
        residual_mode = "global_matched"
        device = torch.device("cpu")
        prefix_embeddings = torch.zeros(1, 2, 3)
        residual_mlp = torch.nn.Identity()

    raw = torch.full((4, 3), 4.0)
    # ResidualPromptMLP.apply(Identity, raw) = raw + raw.
    actual = apply_final_residual_with_alpha(Dummy(), raw, alpha=0.25)
    assert torch.equal(actual, raw * 1.25)


def test_global_alpha_selection_uses_macro_and_breaks_ties_downward() -> None:
    from skillopt.softprefix.interventions_v2 import select_global_alpha

    rows = []
    for alpha, scores in ((0.0, (0.4, 0.6)), (0.5, (0.5, 0.5)), (1.0, (0.7, 0.3))):
        for task, score in zip(("a", "b"), scores, strict=True):
            rows.append({
                "family": "alpha",
                "alpha": alpha,
                "target_task": task,
                "score": score,
            })
    selected = select_global_alpha(rows, ["a", "b"])
    # alpha 0 and 0.5 tie at 0.5; the declared rule chooses the smaller.
    assert selected["selected"]["alpha"] == 0.0
    assert selected["selected"]["macro_score"] == 0.5


def test_alpha_parser_rejects_negative_and_deduplicates() -> None:
    from skillopt.softprefix.interventions_v2 import parse_residual_alphas

    assert parse_residual_alphas([0, "0.5", 0.5, 1]) == [0.0, 0.5, 1.0]
    _assert_raises(
        ValueError,
        "non-negative",
        lambda: parse_residual_alphas([0, -0.1]),
    )


if __name__ == "__main__":
    test_functions = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for test_name, test_function in test_functions:
        test_function()
        print(f"PASS {test_name}")
    print(f"{len(test_functions)} intervention tests passed")
