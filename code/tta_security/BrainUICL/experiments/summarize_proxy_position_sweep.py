#!/usr/bin/env python3
"""Validate and summarize the full49 contiguous proxy-position sweep."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METHODS = (("ewc", "EWC"), ("plain_er", "Plain ER"))
SCHEDULES = (
    ("front_k05", "前置 K=5", tuple(range(1, 6))),
    ("back_k05", "后置 K=5", tuple(range(45, 50))),
    ("front_k10", "前置 K=10", tuple(range(1, 11))),
    ("back_k10", "后置 K=10", tuple(range(40, 50))),
    ("front_k25", "前置 K=25", tuple(range(1, 26))),
    ("back_k25", "后置 K=25", tuple(range(25, 50))),
)
ENDPOINTS = ("old_acc", "old_mf1", "new_acc", "new_mf1")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_path(root: Path, method: str, schedule: str, condition: str) -> Path:
    run = root / "runs" / f"{method}_{schedule}_{condition}"
    return run / method / "metrics.json" if method == "ewc" else run / "metrics.json"


def odd_metrics_path(root: Path, method: str, condition: str) -> Path:
    run = root / "runs" / f"{method}_{condition}"
    return run / method / "metrics.json" if method == "ewc" else run / "metrics.json"


def final_view(metrics: dict[str, Any]) -> dict[str, float]:
    summary = metrics["summary"]
    return {
        "old_acc": float(summary["final_old_acc"]),
        "old_mf1": float(summary["final_old_mf1"]),
        "new_acc": float(summary["final_seen_acc"]),
        "new_mf1": float(summary["final_seen_mf1"]),
        "bwt_acc": float(summary["bwt_acc"]),
        "bwt_mf1": float(summary["bwt_mf1"]),
    }


def old_view_at(metrics: dict[str, Any], task: int) -> dict[str, float]:
    row = metrics["tasks"][task - 1]["old_generalization_after"]
    return {"old_acc": float(row["acc"]), "old_mf1": float(row["mf1"])}


def difference(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in right}


def assert_close(left: float, right: float, message: str) -> None:
    if abs(left - right) > 1e-10:
        raise ValueError(f"{message}: {left} != {right}")


def validate_clean_prefix(
    static: dict[str, Any], feedback: dict[str, Any], first_proxy_task: int
) -> None:
    for task_index in range(1, first_proxy_task):
        left = static["tasks"][task_index - 1]
        right = feedback["tasks"][task_index - 1]
        for field in ("current_before", "current_after", "old_generalization_after"):
            for metric in ("acc", "mf1"):
                assert_close(
                    float(left[field][metric]),
                    float(right[field][metric]),
                    f"Pre-proxy task {task_index} {field}.{metric} drifted",
                )


def validate_run(
    metrics: dict[str, Any],
    expected_order: list[int],
    expected_proxy_tasks: tuple[int, ...],
    *,
    mode: str,
) -> dict[str, Any]:
    tasks = metrics["tasks"]
    if len(tasks) != 49 or [int(row["subject"]) for row in tasks] != expected_order:
        raise ValueError("Full49 task order mismatch")
    expected_proxy = set(expected_proxy_tasks)
    expected_clean = set(range(1, 50)) - expected_proxy
    protocol = metrics["protocol"]["progressive_feedback_proxy"]
    if protocol["mode"] != mode:
        raise ValueError(f"Expected proxy mode {mode}, got {protocol['mode']}")
    if set(map(int, protocol["proxy_tasks"])) != expected_proxy:
        raise ValueError("Proxy task schedule mismatch")
    if set(map(int, protocol["clean_feedback_tasks"])) != expected_clean:
        raise ValueError("Clean-feedback task schedule mismatch")
    if protocol["incremental_clean_hard_labels_visible"] is not False:
        raise ValueError("Incremental hard labels were exposed")
    if int(protocol["source_labeled_sequences"]) != 1030:
        raise ValueError("Unexpected source labeled-buffer size")
    if not protocol["upload_full_pool"] or int(protocol["base_sequences"]) != 48:
        raise ValueError("The full 48-sequence upload protocol was not used")
    rows = [row["progressive_proxy"] for row in tasks]
    proxy_rows = [row for row in rows if row and row.get("feedback_kind") == "proxy"]
    clean_rows = [row for row in rows if row and row.get("feedback_kind") == "clean"]
    if len(proxy_rows) != len(expected_proxy) or len(clean_rows) != len(expected_clean):
        raise ValueError("Proxy/clean role count mismatch")
    if any(int(row["proxy_sequences"]) != 48 for row in proxy_rows):
        raise ValueError("A proxy task did not upload all 48 sequences")
    if any(int(row["labeled_buffer_sequences"]) != 1030 for row in rows):
        raise ValueError("Incremental data entered the source hard-label buffer")
    clean_sequences = sum(int(row["feedback_sequences"]) for row in clean_rows)
    result: dict[str, Any] = {
        "proxy_tasks": len(proxy_rows),
        "proxy_sequences": 48 * len(proxy_rows),
        "clean_feedback_tasks": len(clean_rows),
        "clean_sequences": clean_sequences,
        "stream_sequences": 48 * len(proxy_rows) + clean_sequences,
    }
    if mode == "feedback":
        source_cosines = [float(row["source_gradient_cosine"]) for row in proxy_rows]
        gate = float(metrics["config"]["progressive_max_source_gradient_cosine"])
        if any(not bool(row["source_conflict_accepted"]) for row in proxy_rows):
            raise ValueError("A proxy task failed the source-gradient gate")
        if max(source_cosines) > gate + 1e-7:
            raise ValueError("A proxy task exceeded the source-gradient gate")
        max_step = max(float(row["max_step_relative_l2"]) for row in proxy_rows)
        max_cumulative = max(
            float(row["max_cumulative_relative_l2"]) for row in proxy_rows
        )
        if max_step > 0.050001 or max_cumulative > 0.200001:
            raise ValueError("A proxy upload exceeded its relative-L2 budget")
        result.update(
            {
                "source_gradient_cosine_mean": statistics.fmean(source_cosines),
                "source_gradient_cosine_min": min(source_cosines),
                "source_gradient_cosine_max": max(source_cosines),
                "source_gradient_gate": gate,
                "max_step_relative_l2": max_step,
                "max_cumulative_relative_l2": max_cumulative,
                "final_mean_cumulative_relative_l2": float(
                    proxy_rows[-1]["mean_cumulative_relative_l2"]
                ),
                "mean_history_gradient_cosine": (
                    statistics.fmean(
                        float(row["history_gradient_cosine"])
                        for row in proxy_rows[1:]
                    )
                    if len(proxy_rows) > 1
                    else None
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
            }
        )
    return result


def analyze_pair(
    static: dict[str, Any],
    feedback: dict[str, Any],
    order: list[int],
    proxy_tasks: tuple[int, ...],
    method: str,
) -> dict[str, Any]:
    static_validation = validate_run(
        static, order, proxy_tasks, mode="static"
    )
    feedback_validation = validate_run(
        feedback, order, proxy_tasks, mode="feedback"
    )
    validate_clean_prefix(static, feedback, proxy_tasks[0])
    static_final = final_view(static)
    feedback_final = final_view(feedback)
    final_delta = difference(feedback_final, static_final)
    block_end = proxy_tasks[-1]
    static_block = old_view_at(static, block_end)
    feedback_block = old_view_at(feedback, block_end)
    block_delta = difference(feedback_block, static_block)
    row: dict[str, Any] = {
        "proxy_tasks": list(proxy_tasks),
        "block_end_task": block_end,
        "static": static_final,
        "feedback": feedback_final,
        "delta_static": final_delta,
        "all_endpoints_down": all(final_delta[key] < 0.0 for key in ENDPOINTS),
        "block_end_old": {
            "static": static_block,
            "feedback": feedback_block,
            "delta_static": block_delta,
        },
        "post_block_old_delta_change": {
            "old_acc": final_delta["old_acc"] - block_delta["old_acc"],
            "old_mf1": final_delta["old_mf1"] - block_delta["old_mf1"],
        },
        "static_validation": static_validation,
        "feedback_validation": feedback_validation,
    }
    if method == "plain_er":
        row["memory_proxy_fraction"] = float(
            feedback["summary"]["final_memory_poisoned_fraction"]
        )
        row["replay_proxy_fraction"] = float(
            feedback["summary"]["replay_poisoned_fraction"]
        )
    return row


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def pair(row: dict[str, float], prefix: str, formatter) -> str:
    return f"{formatter(row[f'{prefix}_acc'])}/{formatter(row[f'{prefix}_mf1'])}"


def endpoint_pair(row: dict[str, float], formatter) -> str:
    return f"{formatter(row['old_acc'])}/{formatter(row['old_mf1'])}"


def render(report: dict[str, Any]) -> str:
    labels = {name: label for name, label, _tasks in SCHEDULES}
    lines = [
        "## Proxy连续上传位置与数量的Full49验证",
        "",
        "固定seed 4321、原始49-task subject顺序、EWC/Plain ER配置和Proxy v2 5%单步/20%累计预算。连续区块测试K=5、10、25，分别置于任务流最前端或最后端；每个proxy任务完整上传subject 18的48条sequence，其余任务上传原clean个体。每个位置都有完全同位置、同数量的Fixed-clean48对照，因此下表差值只比较数据变化本身。source hard-label buffer固定为1030条pretrain sequence，增量阶段只使用victim返回概率。",
        "",
        "| 连续位置 | 方法 | Fixed old ACC/MF1 | Proxy old ACC/MF1 | Δold | Fixed new ACC/MF1 | Proxy new ACC/MF1 | Δnew | 四终点均下降 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for schedule, label, _tasks in SCHEDULES:
        for method, method_label in METHODS:
            row = report["schedules"][schedule][method]
            lines.append(
                f"| {label} | {method_label} | "
                f"{pair(row['static'], 'old', pct)} | "
                f"{pair(row['feedback'], 'old', pct)} | "
                f"{pair(row['delta_static'], 'old', pp)} | "
                f"{pair(row['static'], 'new', pct)} | "
                f"{pair(row['feedback'], 'new', pct)} | "
                f"{pair(row['delta_static'], 'new', pp)} | "
                f"{'是' if row['all_endpoints_down'] else '否'} |"
            )
    if report.get("odd_reference"):
        for method, method_label in METHODS:
            row = report["odd_reference"][method]
            lines.append(
                f"| 交错 K=25（既有参照） | {method_label} | "
                f"{pair(row['static'], 'old', pct)} | "
                f"{pair(row['feedback'], 'old', pct)} | "
                f"{pair(row['delta_static'], 'old', pp)} | "
                f"{pair(row['static'], 'new', pct)} | "
                f"{pair(row['feedback'], 'new', pct)} | "
                f"{pair(row['delta_static'], 'new', pp)} | "
                f"{'是' if row['all_endpoints_down'] else '否'} |"
            )
    lines.extend(
        [
            "",
            "| K | 方法 | 后置−前置 Δold ACC/MF1 | 后置−前置 Δnew ACC/MF1 | 解释 |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for k in (5, 10, 25):
        for method, method_label in METHODS:
            comparison = report["position_comparisons"][f"k{k:02d}"][method]
            mean_difference = statistics.fmean(
                comparison[key] for key in ENDPOINTS
            )
            interpretation = "后置平均退化更强" if mean_difference < 0 else "前置平均退化更强"
            lines.append(
                f"| {k} | {method_label} | {pair(comparison, 'old', pp)} | "
                f"{pair(comparison, 'new', pp)} | {interpretation} |"
            )
    lines.extend(
        [
            "",
            "| 连续位置 | 方法 | 区块末Δold ACC/MF1 | 后续clean后的Δold变化 | 最终累计L2均值/最大 | source cosine mean/max | Proxy序列/全流 | ER memory/replay proxy |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for schedule, label, _tasks in SCHEDULES:
        for method, method_label in METHODS:
            row = report["schedules"][schedule][method]
            validation = row["feedback_validation"]
            er_text = "-"
            if method == "plain_er":
                er_text = (
                    f"{pct(row['memory_proxy_fraction'])}/"
                    f"{pct(row['replay_proxy_fraction'])}"
                )
            lines.append(
                f"| {label} | {method_label} | "
                f"{endpoint_pair(row['block_end_old']['delta_static'], pp)} | "
                f"{endpoint_pair(row['post_block_old_delta_change'], pp)} | "
                f"{pct(validation['final_mean_cumulative_relative_l2'])}/"
                f"{pct(validation['max_cumulative_relative_l2'])} | "
                f"{validation['source_gradient_cosine_mean']:.3f}/"
                f"{validation['source_gradient_cosine_max']:.3f} | "
                f"{validation['proxy_sequences']}/"
                f"{validation['stream_sequences']} | {er_text} |"
            )
    strongest = report["strongest_schedule_by_method"]
    front_all_down = [
        schedule
        for schedule, _label, _tasks in SCHEDULES
        if schedule.startswith("front_")
        and all(
            report["schedules"][schedule][method]["all_endpoints_down"]
            for method, _method_label in METHODS
        )
    ]
    back_all_improved = all(
        all(
            report["schedules"][schedule][method]["delta_static"][key] > 0.0
            for method, _method_label in METHODS
            for key in ENDPOINTS
        )
        for schedule, _label, _tasks in SCHEDULES
        if schedule.startswith("back_")
    )
    lines.extend(
        [
            "",
            f"按四个终点差值的平均值，EWC当前最强位置是{labels[strongest['ewc']]}，Plain ER当前最强位置是{labels[strongest['plain_er']]}。位置比较是单seed配对结果；后置区块没有后续clean恢复窗口，前置区块则有，因此这里测量的是实际部署时序效应，而不是与时间位置无关的纯交换性。",
            "",
            (
                "本轮没有任何连续前置区块同时使EWC和Plain ER的四个终点全部下降；交错K=25仍是当前唯一同时满足该条件的K=25安排。"
                if not front_all_down
                else "至少一个连续前置区块同时使两种方法的四个终点下降。"
            ),
            (
                "三个后置连续区块在两种方法的四个终点上均高于各自Fixed-clean48对照，说明在本single-seed轨迹中，末端连续上传更像适配性数据扰动，而不是稳定的退化来源。"
                if back_all_improved
                else "后置连续区块的终点方向并不完全一致。"
            ),
            "位置变化同时改变了被替换的自然subject集合；尽管每个条件都有同位置Fixed-clean48配对对照，结论仍需在多个连续窗口和多个seed上复核，不能把这一次前/后比较视为纯位置的普适因果效应。",
            "",
            "负差值表示Proxy v2相对同位置Fixed-clean48退化；‘后置−前置’为两条配对差值再次相减，负值表示后置更不利。Plain ER的memory/replay比例统计的是被替换记录占比，不代表标签被直接修改。",
        ]
    )
    return "\n".join(lines) + "\n"


def update_bci(path: Path, markdown: str) -> None:
    start = "<!-- PROXY_POSITION_SWEEP_V2_START -->"
    end = "<!-- PROXY_POSITION_SWEEP_V2_END -->"
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
    parser.add_argument("--odd-root", type=Path, default=None)
    parser.add_argument("--bci", type=Path, default=None)
    args = parser.parse_args()
    root = args.run_root.resolve()
    first = load_json(metrics_path(root, "ewc", "front_k05", "static48"))
    order = [int(row["subject"]) for row in first["tasks"]]
    schedules: dict[str, Any] = {}
    for schedule, label, proxy_tasks in SCHEDULES:
        schedule_row: dict[str, Any] = {"label": label, "methods": {}}
        for method, _method_label in METHODS:
            static = load_json(metrics_path(root, method, schedule, "static48"))
            feedback = load_json(metrics_path(root, method, schedule, "v2"))
            schedule_row["methods"][method] = analyze_pair(
                static, feedback, order, proxy_tasks, method
            )
        schedules[schedule] = schedule_row
    report_schedules = {
        schedule: row["methods"] for schedule, row in schedules.items()
    }
    position_comparisons: dict[str, Any] = {}
    for k in (5, 10, 25):
        front = f"front_k{k:02d}"
        back = f"back_k{k:02d}"
        position_comparisons[f"k{k:02d}"] = {
            method: difference(
                report_schedules[back][method]["delta_static"],
                report_schedules[front][method]["delta_static"],
            )
            for method, _method_label in METHODS
        }
    strongest: dict[str, str] = {}
    for method, _method_label in METHODS:
        strongest[method] = min(
            report_schedules,
            key=lambda schedule: statistics.fmean(
                report_schedules[schedule][method]["delta_static"][key]
                for key in ENDPOINTS
            ),
        )
    odd_reference = None
    if args.odd_root is not None:
        odd_root = args.odd_root.resolve()
        odd_tasks = tuple(range(1, 50, 2))
        odd_reference = {}
        for method, _method_label in METHODS:
            static = load_json(odd_metrics_path(odd_root, method, "static48"))
            feedback = load_json(odd_metrics_path(odd_root, method, "v2_s05_c20"))
            odd_reference[method] = analyze_pair(
                static, feedback, order, odd_tasks, method
            )
    report = {
        "schema": "brainuicl-proxy-position-sweep-v2",
        "seed": 4321,
        "task_order": order,
        "schedules": report_schedules,
        "position_comparisons": position_comparisons,
        "strongest_schedule_by_method": strongest,
        "odd_reference": odd_reference,
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
