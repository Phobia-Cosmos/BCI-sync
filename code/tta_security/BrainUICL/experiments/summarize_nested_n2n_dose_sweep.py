#!/usr/bin/env python3
"""Aggregate the nested frozen-proxy dose sweep for EWC and Plain ER."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from derive_nested_n2n_sweep import STAGES
from summarize_canonical_n2n_matrix import validate_paired_prefix


METHODS = (
    ("ewc", "EWC", "regularization"),
    ("plain_er", "Plain ER", "replay"),
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def clean_metrics_path(root: Path, family: str, method: str) -> Path:
    if family == "regularization":
        return root / "runs" / family / method / "clean" / method / "metrics.json"
    return root / "runs" / family / "clean" / "metrics.json"


def shared_metrics_path(stage_root: Path, family: str, method: str) -> Path:
    if family == "regularization":
        return (
            stage_root
            / "runs"
            / family
            / method
            / "shared_proxy"
            / method
            / "metrics.json"
        )
    return stage_root / "runs" / family / "shared_proxy" / "metrics.json"


def final_view(metrics: dict[str, Any]) -> dict[str, float]:
    old = metrics["final"]["old_generalization"]
    summary = metrics["summary"]
    return {
        "old_acc": float(old["acc"]),
        "old_mf1": float(old["mf1"]),
        "new_acc": float(summary["final_seen_acc"]),
        "new_mf1": float(summary["final_seen_mf1"]),
        "bwt_acc": float(summary["bwt_acc"]),
        "bwt_mf1": float(summary["bwt_mf1"]),
    }


def affected_view(
    clean: dict[str, Any],
    shared: dict[str, Any],
    affected_tasks: list[int],
) -> dict[str, Any]:
    rows = []
    for task in affected_tasks:
        clean_row = clean["tasks"][task - 1]
        shared_row = shared["tasks"][task - 1]
        rows.append(
            {
                "task": task,
                "subject": int(shared_row["subject"]),
                "current_after_acc": float(shared_row["current_after"]["acc"])
                - float(clean_row["current_after"]["acc"]),
                "old_after_acc": float(shared_row["old_generalization_after"]["acc"])
                - float(clean_row["old_generalization_after"]["acc"]),
                "pseudo_acc": float(shared_row["pseudo_labels"]["acc_diagnostic_only"])
                - float(clean_row["pseudo_labels"]["acc_diagnostic_only"]),
            }
        )
    return {
        "tasks": rows,
        "mean_current_after_acc": statistics.fmean(
            row["current_after_acc"] for row in rows
        ),
        "mean_old_after_acc": statistics.fmean(row["old_after_acc"] for row in rows),
        "mean_pseudo_acc": statistics.fmean(row["pseudo_acc"] for row in rows),
        "min_current_after_acc": min(row["current_after_acc"] for row in rows),
        "max_current_after_acc": max(row["current_after_acc"] for row in rows),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def render(report: dict[str, Any]) -> str:
    strongest_ewc = min(
        report["stages"],
        key=lambda stage: stage["methods"]["ewc"]["delta"]["old_acc"]
        + stage["methods"]["ewc"]["delta"]["new_acc"],
    )
    strongest_er = min(
        report["stages"],
        key=lambda stage: stage["methods"]["plain_er"]["delta"]["old_acc"]
        + stage["methods"]["plain_er"]["delta"]["new_acc"],
    )
    maximum = report["stages"][-1]
    lines = [
        "## EWC / Plain ER 嵌套 N→N 强度实验",
        "",
        "固定条件：seed 4321，同一 clean Finetune surrogate 生成一次最大 frozen payload；所有阶段共享相同 EEG/EOG 数组并使用严格嵌套的 task 与 sequence mask。每条 relative-L2≤20%、L∞/std≤0.50、5 步，N→N、repeat=0、标签不变。Clean 基线只复用、不重复训练。",
        "",
        "| 阶段 | K/q | Proxy/全流 | EWC Δold ACC/MF1 | EWC Δnew ACC/MF1 | Plain ER Δold ACC/MF1 | Plain ER Δnew ACC/MF1 | ER memory/replay proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report["stages"]:
        ewc = stage["methods"]["ewc"]
        er = stage["methods"]["plain_er"]
        lines.append(
            f"| {stage['stage']} | {stage['task_count']}/{100.0 * stage['nominal_sequence_fraction']:.0f}% | "
            f"{stage['proxy_sequences']}/{stage['uploaded_sequences']} ({100.0 * stage['stream_proxy_fraction']:.2f}%) | "
            f"{pp(ewc['delta']['old_acc'])}/{pp(ewc['delta']['old_mf1'])} | "
            f"{pp(ewc['delta']['new_acc'])}/{pp(ewc['delta']['new_mf1'])} | "
            f"{pp(er['delta']['old_acc'])}/{pp(er['delta']['old_mf1'])} | "
            f"{pp(er['delta']['new_acc'])}/{pp(er['delta']['new_mf1'])} | "
            f"{pct(er['memory_proxy_fraction'])}/{pct(er['replay_proxy_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "| 阶段 | EWC 受影响 task 平均 Δcurrent/Δpseudo ACC | Plain ER 受影响 task 平均 Δcurrent/Δpseudo ACC |",
            "|---|---:|---:|",
        ]
    )
    for stage in report["stages"]:
        ewc = stage["methods"]["ewc"]["affected"]
        er = stage["methods"]["plain_er"]["affected"]
        lines.append(
            f"| {stage['stage']} | {pp(ewc['mean_current_after_acc'])}/{pp(ewc['mean_pseudo_acc'])} | "
            f"{pp(er['mean_current_after_acc'])}/{pp(er['mean_pseudo_acc'])} |"
        )
    lines.extend(
        [
            "",
            f"在已测试档位中，EWC 和 Plain ER 的 old/new ACC 合计退化都在 `{strongest_ewc['stage']}` 达到最大（Plain ER 对应 `{strongest_er['stage']}`）。继续提高到 `{maximum['stage']}` 后，尽管受影响 task 的 pseudo ACC 下降更大，最终退化却缩小，Plain ER 的 old ACC 甚至提高。这说明当前 frozen proxy 的覆盖率-退化关系不是单调函数；全量替换可能形成更一致的域偏移或数据增强，而 50% clean/proxy 混合造成的任务内冲突更强。该机制解释仍需后续消融确认。",
            "",
            "负差值表示 shared proxy 噪声相对 clean 退化。每个阶段都验证完整 49-task 轨迹，并要求首次受影响 task 之前与 clean 逐字段一致。结果为单 seed 强度曲线，不等同于统计显著性。",
        ]
    )
    return "\n".join(lines) + "\n"


def update_bci(path: Path, markdown: str) -> None:
    start = "<!-- N2N_NESTED_DOSE_SWEEP_START -->"
    end = "<!-- N2N_NESTED_DOSE_SWEEP_END -->"
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
    parser.add_argument("--bci", type=Path, default=Path("/home/undefined/Desktop/IPhone/BCI.md"))
    args = parser.parse_args()
    root = args.run_root.resolve()
    clean_root = args.clean_root.resolve()
    sweep = load_json(root / "SWEEP_MANIFESTS.json")
    declared = {row["stage"]: row for row in sweep["stages"]}
    clean_metrics = {
        method: load_json(clean_metrics_path(clean_root, family, method))
        for method, _label, family in METHODS
    }

    stage_rows = []
    for stage_name, task_count, fraction in STAGES:
        stage = declared[stage_name]
        if stage["task_count"] != task_count or not abs(
            stage["nominal_sequence_fraction"] - fraction
        ) < 1e-12:
            raise ValueError(f"Stage metadata mismatch: {stage_name}")
        affected_tasks = [int(task) for task in stage["affected_tasks"]]
        first_affected = min(affected_tasks)
        methods: dict[str, Any] = {}
        for method, label, family in METHODS:
            shared_path = shared_metrics_path(root / "stages" / stage_name, family, method)
            shared_metrics = load_json(shared_path)
            clean = clean_metrics[method]
            prefix = validate_paired_prefix(
                clean,
                shared_metrics,
                affected_task=first_affected,
                expected_tasks=49,
                label=f"{stage_name}/{label}",
            )
            clean_final = final_view(clean)
            shared_final = final_view(shared_metrics)
            method_row = {
                "label": label,
                "clean_metrics": str(clean_metrics_path(clean_root, family, method)),
                "shared_metrics": str(shared_path),
                "clean": clean_final,
                "shared_proxy": shared_final,
                "delta": {
                    key: shared_final[key] - clean_final[key] for key in clean_final
                },
                "affected": affected_view(clean, shared_metrics, affected_tasks),
                "paired_prefix_validation": prefix,
            }
            if method == "plain_er":
                summary = shared_metrics["summary"]
                method_row["memory_proxy_fraction"] = float(
                    summary["final_memory_poisoned_fraction"]
                )
                method_row["replay_proxy_fraction"] = float(
                    summary["replay_poisoned_fraction"]
                )
            methods[method] = method_row
        stage_rows.append({**stage, "methods": methods})

    report = {
        "schema": "brainuicl-nested-n2n-dose-results-v1",
        "protocol": {
            "seed": 4321,
            "n_to_n": True,
            "repeat": 0,
            "max_relative_l2": 0.20,
            "max_linf_over_std": 0.50,
            "optimization_steps": 5,
            "maximum_manifest": sweep["maximum_manifest"],
            "maximum_manifest_sha256": sweep["maximum_manifest_sha256"],
            "clean_root": str(clean_root),
        },
        "stages": stage_rows,
    }
    (root / "FULL_RESULTS.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = render(report)
    (root / "FULL_RESULTS_ZH.md").write_text(markdown, encoding="utf-8")
    update_bci(args.bci.resolve(), markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
