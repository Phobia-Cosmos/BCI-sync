"""Summarize the order-robust Proxy probe against the completed matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynamic_proxy_position_schedule import schedule


DATASETS = {"ISRUC": 49, "FACED": 61}
PLACEMENTS = ("front", "middle", "tail")
METHODS = ("ewc", "full_spr")
SUMMARY_KEYS = (
    "final_old_acc",
    "final_old_mf1",
    "final_seen_acc",
    "final_seen_mf1",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def point_delta(metrics: dict, clean: dict, task: int) -> dict[str, float]:
    current = metrics["tasks"][task - 1]["old_generalization_after"]
    baseline = clean["tasks"][task - 1]["old_generalization_after"]
    return {
        "old_acc": 100.0 * (float(current["acc"]) - float(baseline["acc"])),
        "old_mf1": 100.0 * (float(current["mf1"]) - float(baseline["mf1"])),
    }


def row(
    metrics: dict,
    clean: dict,
    matrix_rows: dict[tuple[str, str], list[dict]],
    dataset: str,
    method: str,
    placement: str,
) -> dict[str, Any]:
    expected = schedule(dataset, 10, placement)
    final_delta = {
        key: 100.0 * (float(metrics["summary"][key]) - float(clean["summary"][key]))
        for key in SUMMARY_KEYS
    }
    window_delta = point_delta(metrics, clean, expected["window_end"])
    old_acc = [
        point_delta(metrics, clean, task)["old_acc"]
        for task in range(1, expected["total_tasks"] + 1)
    ]
    old_mf1 = [
        point_delta(metrics, clean, task)["old_mf1"]
        for task in range(1, expected["total_tasks"] + 1)
    ]
    current_matrix = matrix_rows[(dataset, method)]
    current_row = next(
        row
        for row in current_matrix
        if row["strength_proxy_tasks"] == 10 and row["placement"] == placement
    )
    return {
        "dataset": dataset,
        "method": method,
        "placement": placement,
        "proxy_tasks": expected["proxy_tasks"],
        "proxy_sequences": current_row["proxy_sequences"],
        "final_delta_pp": final_delta,
        "window_end_old_delta_pp": window_delta,
        "worst_old_delta_pp": {
            "acc": min(old_acc),
            "acc_task": old_acc.index(min(old_acc)) + 1,
            "mf1": min(old_mf1),
            "mf1_task": old_mf1.index(min(old_mf1)) + 1,
        },
        "persistence_ratio": {
            "old_acc": (
                final_delta["final_old_acc"] / window_delta["old_acc"]
                if abs(window_delta["old_acc"]) > 1e-9
                else None
            ),
            "old_mf1": (
                final_delta["final_old_mf1"] / window_delta["old_mf1"]
                if abs(window_delta["old_mf1"]) > 1e-9
                else None
            ),
        },
        "baseline_current_final_delta_pp": current_row["final_delta_pp"],
        "survival_protocol": metrics["protocol"]["progressive_feedback_proxy"][
            "survival_objective"
        ],
        "proxy_summary": metrics["final"]["progressive_proxy"],
        "memory": {
            key: metrics["summary"].get(key)
            for key in (
                "final_memory_proxy_epoch_fraction",
                "proxy_replay_fraction",
            )
            if key in metrics["summary"]
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--completed-matrix", type=Path, required=True)
    parser.add_argument("--regularization-clean-isruc", type=Path, required=True)
    parser.add_argument("--regularization-clean-faced", type=Path, required=True)
    parser.add_argument("--replay-clean-isruc", type=Path, required=True)
    parser.add_argument("--replay-clean-faced", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = read(args.completed_matrix)["rows"]
    matrix_rows: dict[tuple[str, str], list[dict]] = {}
    for dataset in DATASETS:
        for method in METHODS:
            matrix_rows[(dataset, method)] = [
                item
                for item in matrix
                if item["dataset"] == dataset and item["method"] == method
            ]
    clean_paths = {
        ("ISRUC", "ewc"): args.regularization_clean_isruc,
        ("FACED", "ewc"): args.regularization_clean_faced,
        ("ISRUC", "full_spr"): args.replay_clean_isruc,
        ("FACED", "full_spr"): args.replay_clean_faced,
    }
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for placement in PLACEMENTS:
            for method in METHODS:
                root = args.run_root / "runs" / dataset.lower() / f"k10_{placement}"
                metrics_path = (
                    root / "regularization" / "ewc" / "metrics.json"
                    if method == "ewc"
                    else root / "full_spr" / "metrics.json"
                )
                metrics = read(metrics_path)
                clean = read(clean_paths[(dataset, method)])
                rows.append(
                    row(
                        metrics,
                        clean,
                        matrix_rows,
                        dataset,
                        method,
                        placement,
                    )
                )
    payload = {
        "protocol": {
            "strength": 10,
            "placements": list(PLACEMENTS),
            "methods": list(METHODS),
            "future_incremental_data_visible": False,
            "repair_data": "source_pretrain_hard_labels_only",
            "paired_baseline": "same-method natural clean",
        },
        "rows": rows,
    }
    (args.run_root / "RESULTS.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"validated {len(rows)} order-probe rows")


if __name__ == "__main__":
    main()
