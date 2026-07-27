#!/usr/bin/env python3
"""Validate and summarize the cross-CL full-pool feedback proxy experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METHODS = (("ewc", "EWC"), ("plain_er", "Plain ER"))
CONDITIONS = (
    ("v2_s05_c20", "Proxy v2 5%/20%"),
    ("v2_s10_c40", "Proxy v2 10%/40%"),
    ("v2_s15_c60", "Proxy v2 15%/60%"),
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_path(root: Path, method: str, condition: str) -> Path:
    base = root / "runs" / f"{method}_{condition}"
    return base / method / "metrics.json" if method == "ewc" else base / "metrics.json"


def clean_path(root: Path, method: str) -> Path:
    if method == "ewc":
        return root / "runs" / "regularization" / "ewc" / "clean" / "ewc" / "metrics.json"
    return root / "runs" / "replay" / "clean" / "metrics.json"


def view(metrics: dict[str, Any]) -> dict[str, float]:
    summary = metrics["summary"]
    return {
        "old_acc": float(summary["final_old_acc"]),
        "old_mf1": float(summary["final_old_mf1"]),
        "new_acc": float(summary["final_seen_acc"]),
        "new_mf1": float(summary["final_seen_mf1"]),
        "bwt_acc": float(summary["bwt_acc"]),
        "bwt_mf1": float(summary["bwt_mf1"]),
    }


def difference(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in right}


def validate(
    metrics: dict[str, Any],
    expected_order: list[int],
    *,
    feedback: bool,
) -> dict[str, Any]:
    tasks = metrics["tasks"]
    if len(tasks) != 49 or [int(row["subject"]) for row in tasks] != expected_order:
        raise ValueError("Full49 task order mismatch")
    protocol = metrics["protocol"]["progressive_feedback_proxy"]
    if protocol["incremental_clean_hard_labels_visible"] is not False:
        raise ValueError("Incremental clean hard labels were exposed")
    if protocol["source_labeled_sequences"] != 1030:
        raise ValueError("Unexpected source labeled-buffer size")
    if not protocol["upload_full_pool"] or protocol["base_sequences"] != 48:
        raise ValueError("The fixed 48-sequence upload protocol was not used")
    rows = [row["progressive_proxy"] for row in tasks]
    proxy_rows = [row for row in rows if row and row.get("feedback_kind") == "proxy"]
    clean_rows = [row for row in rows if row and row.get("feedback_kind") == "clean"]
    if len(proxy_rows) != 25 or len(clean_rows) != 24:
        raise ValueError("Odd/even role count mismatch")
    if any(int(row["proxy_sequences"]) != 48 for row in proxy_rows):
        raise ValueError("A proxy task did not upload all 48 sequences")
    if any(int(row["labeled_buffer_sequences"]) != 1030 for row in rows):
        raise ValueError("Incremental data entered the hard-label buffer")
    result = {
        "tasks": 49,
        "proxy_tasks": 25,
        "clean_feedback_tasks": 24,
        "proxy_sequences": 1200,
        "clean_sequences": sum(int(row["feedback_sequences"]) for row in clean_rows),
        "feedback_sequences": int(rows[-1]["feedback_buffer_sequences"]),
        "labeled_sequences": int(rows[-1]["labeled_buffer_sequences"]),
    }
    if feedback:
        source_cosines = [float(row["source_gradient_cosine"]) for row in proxy_rows]
        if any(not bool(row["source_conflict_accepted"]) for row in proxy_rows):
            raise ValueError("A proxy task failed the source-conflict gate")
        source_gate = float(
            metrics["config"]["progressive_max_source_gradient_cosine"]
        )
        if max(source_cosines) > source_gate + 1e-7:
            raise ValueError("A proxy task exceeded the source-gradient gate")
        result.update(
            {
                "source_gradient_cosine_mean": statistics.fmean(source_cosines),
                "source_gradient_cosine_max": max(source_cosines),
                "source_gradient_cosine_min": min(source_cosines),
                "source_gradient_gate": source_gate,
                "max_step_relative_l2": max(
                    float(row["max_step_relative_l2"]) for row in proxy_rows
                ),
                "max_cumulative_relative_l2": max(
                    float(row["max_cumulative_relative_l2"]) for row in proxy_rows
                ),
                "mean_final_cumulative_relative_l2": float(
                    proxy_rows[-1]["mean_cumulative_relative_l2"]
                ),
                "mean_input_direction_cosine": statistics.fmean(
                    float(row["input_direction_cosine"])
                    for row in proxy_rows[1:]
                ),
                "mean_pseudo_label_preservation": statistics.fmean(
                    float(row["generation"]["pseudo_label_preservation"])
                    for row in proxy_rows
                ),
                "mean_target_hit_rate": statistics.fmean(
                    float(row["generation"]["target_hit_rate"])
                    for row in proxy_rows
                ),
                "kl_improved_events": sum(
                    float(row["response_kl_after"])
                    < float(row["response_kl_before"])
                    for row in rows
                ),
                "mean_response_kl_before": statistics.fmean(
                    float(row["response_kl_before"]) for row in rows
                ),
                "mean_response_kl_after": statistics.fmean(
                    float(row["response_kl_after"]) for row in rows
                ),
                "mean_response_kl_delta": statistics.fmean(
                    float(row["response_kl_after"])
                    - float(row["response_kl_before"])
                    for row in rows
                ),
            }
        )
    return result


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def pair(row: dict[str, float], prefix: str, formatter) -> str:
    return f"{formatter(row[f'{prefix}_acc'])}/{formatter(row[f'{prefix}_mf1'])}"


def render(report: dict[str, Any]) -> str:
    minimum = report["minimum_effective_condition"]
    minimum_label = dict(CONDITIONS)[minimum]
    lines = [
        "## 跨正则化与Replay的Full-pool Proxy v2验证",
        "",
        "协议固定为seed 4321和原始49-task顺序。奇数25个位置每次完整上传subject 18的48条sequence，偶数24个位置保留原clean上传；Fixed-clean48 control在奇数位置上传未变化的同一48条clean，因此v2与control的任务数、位置和数据量完全一致。pretrain训练集1030条clean及hard label是本地唯一hard-label监督；后续clean和proxy都只保存victim返回的五类概率，不能读取增量阶段annotation。",
        "",
        "v2每次从上一版48条继续变化，实际单步relative-L2不超过5%、累计不超过20%，输入方向锥余弦不低于约0.98；本地proxy每次使用source CE和历史clean/proxy概率KL更新，并在最新proxy参数下重算历史gradient。只有任务级proxy梯度与64条source样本聚合梯度的余弦不大于0.10时才允许上传，用低冲突或近正交更新排除旧版约+0.79的正常同向更新。EWC与Plain ER使用完全相同的生成规则和超参数。",
        "",
        "| 方法/条件 | old ACC/MF1 | Δold vs Fixed-clean48 | new ACC/MF1 | Δnew vs Fixed-clean48 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in METHODS:
        row = report["methods"][method]
        static = row["static48"]["final"]
        v2 = row["conditions"][minimum]["final"]
        lines.append(
            f"| {label} / Fixed-clean48 | {pair(static, 'old', pct)} | +0.00 pp/+0.00 pp | "
            f"{pair(static, 'new', pct)} | +0.00 pp/+0.00 pp |"
        )
        lines.append(
            f"| {label} / {minimum_label} | {pair(v2, 'old', pct)} | "
            f"{pair(row['conditions'][minimum]['delta_static'], 'old', pp)} | "
            f"{pair(v2, 'new', pct)} | "
            f"{pair(row['conditions'][minimum]['delta_static'], 'new', pp)} |"
        )
    lines.extend(
        [
            "",
            "| 方法/强度 | old ACC/MF1 | Δold vs Fixed-clean48 | new ACC/MF1 | Δnew vs Fixed-clean48 | 主判定 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, label in METHODS:
        row = report["methods"][method]
        for condition, condition_label in CONDITIONS:
            condition_row = row["conditions"][condition]
            lines.append(
                f"| {label} / {condition_label} | "
                f"{pair(condition_row['final'], 'old', pct)} | "
                f"{pair(condition_row['delta_static'], 'old', pp)} | "
                f"{pair(condition_row['final'], 'new', pct)} | "
                f"{pair(condition_row['delta_static'], 'new', pp)} | "
                f"{'通过' if condition_row['success'] else '未通过'} |"
            )
    lines.extend(
        [
            "",
            "| 方法/强度 | Δold ACC/MF1 vs原始Clean49 | Δnew ACC/MF1 vs原始Clean49 | source-gradient cosine mean/max/gate | 最终平均/最大累计relative-L2 | KL before→after/mean Δ | ER memory/replay proxy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, label in METHODS:
        row = report["methods"][method]
        condition_row = row["conditions"][minimum]
        validation = condition_row["validation"]
        er_text = "-"
        if method == "plain_er":
            er_text = (
                f"{pct(condition_row['memory_proxy_fraction'])}/"
                f"{pct(condition_row['replay_proxy_fraction'])}"
            )
        lines.append(
            f"| {label} / {minimum_label} | {pair(condition_row['delta_clean'], 'old', pp)} | "
            f"{pair(condition_row['delta_clean'], 'new', pp)} | "
            f"{validation['source_gradient_cosine_mean']:.3f}/"
            f"{validation['source_gradient_cosine_max']:.3f}/"
            f"{validation['source_gradient_gate']:.3f} | "
            f"{pct(validation['mean_final_cumulative_relative_l2'])}/"
            f"{pct(validation['max_cumulative_relative_l2'])} | "
            f"{validation['mean_response_kl_before']:.3f}→"
            f"{validation['mean_response_kl_after']:.3f}/"
            f"{validation['mean_response_kl_delta']:+.3f} "
            f"({validation['kl_improved_events']}/49) | {er_text} |"
        )
    success = bool(report["cross_method_success"])
    lines.extend(
        [
            "",
            (
                f"结果满足主成功条件：{minimum_label} 是同时使EWC和Plain ER的old/new ACC与MF1均低于各自同量Fixed-clean48 control的最低强度。"
                if success
                else "结果未满足跨方法成功条件；至少一个算法或终点没有低于同量Fixed-clean48 control。所有结果仍完整报告，不能只保留有利终点。"
            ),
            "",
            "负差值表示退化。Fixed-clean48是判断proxy数据变化本身是否有效的主对照；原始Clean49的数据身份和奇数任务数据量不同，只作为额外部署参照，不参与主成功判定。结果是单seed点估计，尚不代表跨seed稳定性。",
        ]
    )
    return "\n".join(lines) + "\n"


def update_bci(path: Path, markdown: str) -> None:
    start = "<!-- CROSS_CL_PROXY_V2_START -->"
    end = "<!-- CROSS_CL_PROXY_V2_END -->"
    text = path.read_text(encoding="utf-8")
    block = f"{start}\n{markdown}{end}"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--bci", type=Path, default=None)
    args = parser.parse_args()
    root = args.run_root.resolve()
    clean_root = args.clean_root.resolve()
    split = load_json(root / "runs" / "ewc_v2_s05_c20" / "split.json")
    order = [int(subject) for subject in split["new_order"]]
    if len(order) != 49:
        raise ValueError("Expected full49 split")
    endpoint_keys = ("old_acc", "old_mf1", "new_acc", "new_mf1")
    methods: dict[str, Any] = {}
    for method, label in METHODS:
        clean_metrics = load_json(clean_path(clean_root, method))
        static_metrics = load_json(run_path(root, method, "static48"))
        clean_final = view(clean_metrics)
        static_final = view(static_metrics)
        row: dict[str, Any] = {
            "label": label,
            "clean": clean_final,
            "static48": {
                "final": static_final,
                "delta_clean": difference(static_final, clean_final),
                "validation": validate(static_metrics, order, feedback=False),
            },
            "conditions": {},
        }
        for condition, _condition_label in CONDITIONS:
            condition_metrics = load_json(run_path(root, method, condition))
            condition_final = view(condition_metrics)
            delta_static = difference(condition_final, static_final)
            condition_row: dict[str, Any] = {
                "final": condition_final,
                "delta_static": delta_static,
                "delta_clean": difference(condition_final, clean_final),
                "validation": validate(condition_metrics, order, feedback=True),
                "success": all(delta_static[key] < 0.0 for key in endpoint_keys),
            }
            if method == "plain_er":
                condition_row["memory_proxy_fraction"] = float(
                    condition_metrics["summary"]["final_memory_poisoned_fraction"]
                )
                condition_row["replay_proxy_fraction"] = float(
                    condition_metrics["summary"]["replay_poisoned_fraction"]
                )
            row["conditions"][condition] = condition_row
        methods[method] = row
    success_by_condition = {
        condition: all(
            methods[method]["conditions"][condition]["success"]
            for method, _label in METHODS
        )
        for condition, _condition_label in CONDITIONS
    }
    minimum_effective_condition = next(
        (
            condition
            for condition, _condition_label in CONDITIONS
            if success_by_condition[condition]
        ),
        None,
    )
    success = minimum_effective_condition is not None
    report = {
        "schema": "brainuicl-cross-cl-full-pool-proxy-v2",
        "cross_method_success": success,
        "success_by_condition": success_by_condition,
        "minimum_effective_condition": minimum_effective_condition,
        "methods": methods,
    }
    (root / "FULL_RESULTS.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = render(report)
    (root / "FULL_RESULTS_ZH.md").write_text(markdown, encoding="utf-8")
    if args.bci is not None:
        update_bci(args.bci.resolve(), markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
