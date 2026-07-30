#!/usr/bin/env python3
"""Validate and summarize the dynamic Proxy count-by-position matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynamic_proxy_position_schedule import (
    DATASET_TASKS,
    PLACEMENTS,
    STRENGTHS,
    schedule,
)


REGULARIZATION_METHODS = ("finetune", "ewc", "online_ewc", "si", "mas")
FULL_METHODS = ("full_spr", "full_puridiver")
METHODS = REGULARIZATION_METHODS + FULL_METHODS
DISPLAY_NAMES = {
    "finetune": "Finetune",
    "ewc": "EWC",
    "online_ewc": "Online EWC",
    "si": "SI",
    "mas": "MAS",
    "full_spr": "Full SPR-EEG",
    "full_puridiver": "Full PuriDivER-EEG",
}
SUMMARY_KEYS = (
    "final_old_acc",
    "final_old_mf1",
    "final_seen_acc",
    "final_seen_mf1",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_task_spec(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def clean_metrics_path(
    args: argparse.Namespace, dataset: str, method: str
) -> Path:
    if method in REGULARIZATION_METHODS:
        root = (
            args.regularization_clean_isruc
            if dataset == "ISRUC"
            else args.regularization_clean_faced
        )
        return root / method / "metrics.json"
    return args.run_root / "clean" / dataset.lower() / method / "metrics.json"


def proxy_metrics_path(
    run_root: Path,
    dataset: str,
    strength: int,
    placement: str,
    method: str,
) -> Path:
    base = (
        run_root
        / "runs"
        / dataset.lower()
        / f"k{strength}_{placement}"
    )
    if method in REGULARIZATION_METHODS:
        return base / "regularization" / method / "metrics.json"
    return base / method / "metrics.json"


def validate_clean(
    metrics: dict[str, Any], dataset: str, method: str, total_tasks: int
) -> None:
    config = metrics["config"]
    if config["dataset"] != dataset:
        raise ValueError(f"{method} clean dataset mismatch")
    if bool(config.get("freeze_bn_stats")):
        raise ValueError(f"{dataset}/{method} clean unexpectedly freezes BN")
    if config.get("progressive_proxy_mode", "none") != "none":
        raise ValueError(f"{dataset}/{method} clean contains a Proxy stream")
    if len(metrics["tasks"]) != total_tasks:
        raise ValueError(f"{dataset}/{method} clean task count mismatch")
    missing = [key for key in SUMMARY_KEYS if key not in metrics["summary"]]
    if missing:
        raise ValueError(f"{dataset}/{method} clean missing metrics: {missing}")


def validate_proxy(
    metrics: dict[str, Any],
    clean: dict[str, Any],
    dataset: str,
    method: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    config = metrics["config"]
    if config["dataset"] != dataset:
        raise ValueError(f"{dataset}/{method} Proxy dataset mismatch")
    if bool(config.get("freeze_bn_stats")):
        raise ValueError(f"{dataset}/{method} Proxy unexpectedly freezes BN")
    if config.get("progressive_proxy_mode") != "feedback":
        raise ValueError(f"{dataset}/{method} is not a feedback Proxy run")
    if parse_task_spec(config["progressive_proxy_tasks"]) != expected["proxy_tasks"]:
        raise ValueError(f"{dataset}/{method} Proxy task schedule mismatch")
    if (
        parse_task_spec(config["progressive_clean_feedback_tasks"])
        != expected["clean_feedback_tasks"]
    ):
        raise ValueError(f"{dataset}/{method} clean feedback schedule mismatch")
    if not bool(config.get("progressive_match_task_sequence_count")):
        raise ValueError(f"{dataset}/{method} does not match task cardinality")
    if bool(config.get("progressive_upload_full_pool")):
        raise ValueError(f"{dataset}/{method} still uploads a fixed full pool")
    if bool(config.get("progressive_require_source_conflict")):
        raise ValueError(f"{dataset}/{method} unexpectedly uses a hard source gate")
    if not bool(config.get("progressive_require_all_sequences_modified")):
        raise ValueError(f"{dataset}/{method} does not require all sequences modified")
    if float(config.get("progressive_active_fraction", 0.0)) != 1.0:
        raise ValueError(f"{dataset}/{method} does not modify the full upload")

    tasks = metrics["tasks"]
    clean_tasks = clean["tasks"]
    if len(tasks) != expected["total_tasks"] or len(clean_tasks) != len(tasks):
        raise ValueError(f"{dataset}/{method} task count mismatch")
    if [row["subject"] for row in tasks] != [
        row["subject"] for row in clean_tasks
    ]:
        raise ValueError(f"{dataset}/{method} subject order differs from clean")

    proxy_set = set(expected["proxy_tasks"])
    uploaded = 0
    for task_index, (row, clean_row) in enumerate(
        zip(tasks, clean_tasks), start=1
    ):
        proxy_row = row.get("progressive_proxy") or {}
        if task_index in proxy_set:
            clean_epochs = int(clean_row["current_before"]["n_epochs"])
            if clean_epochs % 20:
                raise ValueError(
                    f"{dataset}/{method}/task{task_index} is not 20-epoch sequences"
                )
            expected_sequences = clean_epochs // 20
            if proxy_row.get("kind") != "progressive_proxy":
                raise ValueError(
                    f"{dataset}/{method}/task{task_index} is not a Proxy upload"
                )
            if int(proxy_row.get("proxy_sequences", -1)) != expected_sequences:
                raise ValueError(
                    f"{dataset}/{method}/task{task_index} cardinality mismatch"
                )
            if int(proxy_row.get("unmodified_sequences", -1)) != 0:
                raise ValueError(
                    f"{dataset}/{method}/task{task_index} has unchanged Proxy data"
                )
            uploaded += expected_sequences
        elif proxy_row.get("kind") != "clean_feedback":
            raise ValueError(
                f"{dataset}/{method}/task{task_index} is not original clean feedback"
            )

    final_proxy = metrics["final"]["progressive_proxy"]
    if int(final_proxy["proxy_tasks_completed"]) != expected["strength_proxy_tasks"]:
        raise ValueError(f"{dataset}/{method} completed Proxy count mismatch")
    if int(final_proxy["proxy_sequences_uploaded"]) != uploaded:
        raise ValueError(f"{dataset}/{method} uploaded sequence total mismatch")
    return {"proxy_sequences": uploaded}


def point_deltas(
    metrics: dict[str, Any], clean: dict[str, Any], task_index: int
) -> dict[str, float]:
    current = metrics["tasks"][task_index - 1]["old_generalization_after"]
    baseline = clean["tasks"][task_index - 1]["old_generalization_after"]
    return {
        "old_acc": 100.0 * (float(current["acc"]) - float(baseline["acc"])),
        "old_mf1": 100.0 * (float(current["mf1"]) - float(baseline["mf1"])),
    }


def summarize_row(
    metrics: dict[str, Any],
    clean: dict[str, Any],
    dataset: str,
    method: str,
    expected: dict[str, Any],
    uploaded: int,
) -> dict[str, Any]:
    final_delta = {
        key: 100.0 * (
            float(metrics["summary"][key]) - float(clean["summary"][key])
        )
        for key in SUMMARY_KEYS
    }
    trajectory = [
        point_deltas(metrics, clean, task_index)
        for task_index in range(1, expected["total_tasks"] + 1)
    ]
    worst_acc_task = min(
        range(1, expected["total_tasks"] + 1),
        key=lambda task: trajectory[task - 1]["old_acc"],
    )
    worst_mf1_task = min(
        range(1, expected["total_tasks"] + 1),
        key=lambda task: trajectory[task - 1]["old_mf1"],
    )
    total_sequences = sum(
        int(row["current_before"]["n_epochs"]) // 20 for row in clean["tasks"]
    )
    row = {
        "dataset": dataset,
        "method": method,
        "strength_proxy_tasks": expected["strength_proxy_tasks"],
        "placement": expected["placement"],
        "window_start": expected["window_start"],
        "window_end": expected["window_end"],
        "proxy_tasks": expected["proxy_tasks"],
        "proxy_sequences": uploaded,
        "total_sequences": total_sequences,
        "proxy_sequence_fraction": uploaded / total_sequences,
        "window_end_old_delta_pp": point_deltas(
            metrics, clean, expected["window_end"]
        ),
        "worst_old_delta_pp": {
            "acc": trajectory[worst_acc_task - 1]["old_acc"],
            "acc_task": worst_acc_task,
            "mf1": trajectory[worst_mf1_task - 1]["old_mf1"],
            "mf1_task": worst_mf1_task,
        },
        "final_delta_pp": final_delta,
        "final_absolute": {
            key: float(metrics["summary"][key]) for key in SUMMARY_KEYS
        },
    }
    if method in FULL_METHODS:
        row["memory"] = {
            key: float(metrics["summary"][key])
            for key in (
                "final_memory_proxy_epoch_fraction",
                "proxy_replay_fraction",
            )
        }
    return row


def signed(value: float) -> str:
    return f"{value:+.2f}"


def condition_label(strength: int, placement: str) -> str:
    position = {"front": "前", "middle": "中", "tail": "尾"}[placement]
    return f"K={strength}{position}"


def matrix_table(
    rows: list[dict[str, Any]], dataset: str, key_a: str, key_b: str
) -> list[str]:
    conditions = [
        (strength, placement)
        for strength in STRENGTHS
        for placement in PLACEMENTS
    ]
    lines = [
        "| 方法 | "
        + " | ".join(condition_label(*condition) for condition in conditions)
        + " |",
        "|---|" + "---:|" * len(conditions),
    ]
    index = {
        (row["method"], row["strength_proxy_tasks"], row["placement"]): row
        for row in rows
        if row["dataset"] == dataset
    }
    for method in METHODS:
        cells = []
        for strength, placement in conditions:
            delta = index[(method, strength, placement)]["final_delta_pp"]
            cells.append(f"`{signed(delta[key_a])}/{signed(delta[key_b])}`")
        lines.append(f"| {DISPLAY_NAMES[method]} | " + " | ".join(cells) + " |")
    return lines


def dose_table(rows: list[dict[str, Any]], dataset: str) -> list[str]:
    lines = [
        "| 条件 | Proxy任务窗口 | Proxy sequence/总sequence | 占比 |",
        "|---|---:|---:|---:|",
    ]
    for strength in STRENGTHS:
        for placement in PLACEMENTS:
            candidates = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["strength_proxy_tasks"] == strength
                and row["placement"] == placement
            ]
            uploaded = {row["proxy_sequences"] for row in candidates}
            totals = {row["total_sequences"] for row in candidates}
            if len(uploaded) != 1 or len(totals) != 1:
                raise ValueError(f"{dataset} dose differs across methods")
            sample = candidates[0]
            lines.append(
                f"| {condition_label(strength, placement)} | "
                f"{sample['window_start']}-{sample['window_end']} | "
                f"{sample['proxy_sequences']}/{sample['total_sequences']} | "
                f"{100.0 * sample['proxy_sequence_fraction']:.2f}% |"
            )
    return lines


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# 动态Proxy数量与位置双数据集完整矩阵",
        "",
        "本实验把强度定义为Proxy任务数`K=10/20`，不是修改相对L2；单次相对L2上限固定5%，相对初始clean的累计上限固定20%。每个位置上传数量与该位置自然clean subject严格一致，所有上传sequence均修改，非Proxy位置直接使用各自原始clean，主差值为`动态Proxy - 同方法自然clean`。生成目标保留source-conflict权重20并记录source cosine，但不使用硬拒绝gate，避免把不同CL轨迹下候选是否可行混入位置比较。BN running statistics更新；正则化student使用batch 32、`lr=1e-7`，Full SPR/PuriDivER保留完整memory流程并采用共同的batch 32、student `lr=1e-7`协议。结果均为seed 4321单次运行。",
        "",
        "位置采用交错窗口：长度为`2K-1`的窗口置于任务流前端、正中或尾端，窗口内Proxy与clean反馈交替。负差值表示相对自然clean退化，所有数字单位均为百分点。",
        "",
    ]
    for dataset in DATASET_TASKS:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "### 实际上传剂量",
                "",
                *dose_table(rows, dataset),
                "",
                "### 最终old ACC/MF1变化",
                "",
                *matrix_table(
                    rows, dataset, "final_old_acc", "final_old_mf1"
                ),
                "",
                "### 最终seen-new ACC/MF1变化",
                "",
                *matrix_table(
                    rows, dataset, "final_seen_acc", "final_seen_mf1"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 解释边界",
            "",
            "每条动态Proxy轨迹都根据对应victim返回的概率独立更新，因此不同CL方法、不同位置以及K=10/20之间的Proxy数组不是固定不变的共享输入。自然clean对照保持原始subject顺序与每位置数据量；Proxy分支只在manifest指定位置以subject 18动态数据替换该上传槽位，其他位置的clean数组不变。详细的窗口结束变化、全程最差old变化、最终绝对性能以及SPR/PuriDivER memory与replay中的Proxy比例保存在`RESULTS.json`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--regularization-clean-isruc", type=Path, required=True
    )
    parser.add_argument(
        "--regularization-clean-faced", type=Path, required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    clean_runs: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, total_tasks in DATASET_TASKS.items():
        for method in METHODS:
            clean = read_json(clean_metrics_path(args, dataset, method))
            validate_clean(clean, dataset, method, total_tasks)
            clean_runs[(dataset, method)] = clean

    rows: list[dict[str, Any]] = []
    for dataset in DATASET_TASKS:
        for strength in STRENGTHS:
            for placement in PLACEMENTS:
                expected = schedule(dataset, strength, placement)
                for method in METHODS:
                    clean = clean_runs[(dataset, method)]
                    metrics = read_json(
                        proxy_metrics_path(
                            args.run_root,
                            dataset,
                            strength,
                            placement,
                            method,
                        )
                    )
                    diagnostics = validate_proxy(
                        metrics, clean, dataset, method, expected
                    )
                    rows.append(
                        summarize_row(
                            metrics,
                            clean,
                            dataset,
                            method,
                            expected,
                            diagnostics["proxy_sequences"],
                        )
                    )

    payload = {
        "protocol": {
            "seed": 4321,
            "strength_definition": "number of Proxy tasks",
            "strengths": list(STRENGTHS),
            "placements": list(PLACEMENTS),
            "natural_clean_baseline": True,
            "match_task_sequence_count": True,
            "all_proxy_sequences_modified": True,
            "source_conflict_gate": "diagnostic_only",
            "batch_norm_running_stats": "updated",
            "step_relative_l2": 0.05,
            "cumulative_relative_l2": 0.20,
        },
        "clean_metrics": {
            f"{dataset}/{method}": str(
                clean_metrics_path(args, dataset, method).resolve()
            )
            for dataset in DATASET_TASKS
            for method in METHODS
        },
        "rows": rows,
    }
    results_path = args.run_root / "RESULTS.json"
    results_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(args.run_root / "SUMMARY_ZH.md", payload)
    print(f"validated {len(rows)} method-condition rows")


if __name__ == "__main__":
    main()
