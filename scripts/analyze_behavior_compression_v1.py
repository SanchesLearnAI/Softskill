#!/usr/bin/env python3
"""Compare behavior-compression v1 against the registered trusted baselines."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _task_comparison(
    actual: dict[str, Any],
    baseline: dict[str, Any],
    primary_metric: dict[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_name, metric_name in primary_metric.items():
        current = actual[task_name]
        reference = baseline[task_name]
        primary_value = float(current[metric_name])
        primary_reference = float(reference[metric_name])
        result[task_name] = {
            "primary_metric": metric_name,
            "current": {key: float(value) for key, value in current.items() if key in {"hard", "soft"}},
            "baseline": {key: float(value) for key, value in reference.items() if key in {"hard", "soft"}},
            "absolute_delta": {
                key: float(current[key]) - float(reference[key])
                for key in {"hard", "soft"}
                if key in current and key in reference
            },
            "primary_retention": primary_value / primary_reference,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="outputs/behavior_compression/v1/shared16_task16_raw_two_stage/seed1/summary.json",
    )
    parser.add_argument(
        "--baselines",
        default="skillopt/behavior_compression/v1/baseline_registry.json",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    summary = _load(args.summary)
    baselines = _load(args.baselines)
    if summary.get("layout") != "shared16_task16":
        raise ValueError("behavior v1 summary must use shared16_task16")
    if summary.get("initialization_mode") != "behavior_markdown":
        raise ValueError("behavior v1 summary must use behavior_markdown")
    if summary.get("lr_schedule") != "behavior_two_stage":
        raise ValueError("behavior v1 summary must use behavior_two_stage")
    if summary.get("use_residual_reparameterization") is not False:
        raise ValueError("behavior v1 summary must have residual disabled")

    primary_metric = baselines["primary_metric"]
    independent = _task_comparison(
        summary["test"],
        baselines["independent_32"]["metrics"],
        primary_metric,
    )
    mean_shared = _task_comparison(
        summary["test"],
        baselines["v2_mean_shared_seed1"]["metrics"],
        primary_metric,
    )
    retentions = [record["primary_retention"] for record in independent.values()]
    report = {
        "schema_version": 1,
        "behavior_summary": os.path.abspath(args.summary),
        "baseline_registry": os.path.abspath(args.baselines),
        "comparison_to_independent_32": independent,
        "comparison_to_v2_mean_shared_seed1": mean_shared,
        "normalized_mean_retention_vs_independent_32": sum(retentions) / len(retentions),
        "minimum_task_retention_vs_independent_32": min(retentions),
        "storage": summary["storage"],
        "interpretation_guardrail": (
            "Inspect every task retention and absolute delta, especially LiveMath; "
            "do not substitute an average of incomparable raw task scores."
        ),
    }
    output = args.output or os.path.join(os.path.dirname(args.summary), "comparison.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
