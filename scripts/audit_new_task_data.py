"""Write the stage-1 ChartQA/DROP data and evaluation audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillopt.envs.chartqa.data import (
    chartqa_answer_counts,
    select_chartqa_transfer_splits,
)
from skillopt.envs.drop.data import select_drop_transfer_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa_root", required=True)
    parser.add_argument("--drop_root", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        default=(
            "skillopt/behavior_compression/new_task_transfer/v1/data_audit.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chartqa_splits, chartqa_audit = select_chartqa_transfer_splits(
        args.chartqa_root,
        train_size=None,
        seed=args.seed,
    )
    _, drop_audit = select_drop_transfer_splits(
        args.drop_root,
        train_size=None,
        seed=args.seed,
    )
    chartqa_audit["answer_category_counts"] = {
        split: chartqa_answer_counts(items)
        for split, items in chartqa_splits.items()
    }
    audit = {
        "schema_version": 1,
        "phase": "new_task_transfer_stage1_data_adaptation_and_evaluation",
        "seed": args.seed,
        "tasks": {
            "chartqa": chartqa_audit,
            "drop": drop_audit,
        },
        "excluded_from_this_phase": [
            "F0/F1/I32 implementation",
            "Shared Prefix checkpoint loading",
            "150-step training",
            "HPC job submission",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
