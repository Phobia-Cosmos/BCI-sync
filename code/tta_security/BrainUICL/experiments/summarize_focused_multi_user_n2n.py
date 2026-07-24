#!/usr/bin/env python3
"""Summarize the focused EWC/Plain-ER multi-user N-to-N experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_canonical_n2n_matrix import validate_paired_prefix


METHODS = (
    ("EWC", "regularization", "ewc"),
    ("Plain ER", "replay", "plain_er"),
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_path(root: Path, family: str, method: str, condition: str) -> Path:
    if family == "regularization":
        return root / "runs" / family / method / condition / method / "metrics.json"
    return root / "runs" / family / condition / "metrics.json"


def metric_view(metrics: dict[str, Any]) -> dict[str, float]:
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


def fmt(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def delta_fmt(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def render(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    lines = [
        "## 多用户 N→N 聚焦验证：EWC 与 Plain ER",
        "",
        f"固定条件：seed {protocol['seed']}，受扰动 task {protocol['affected_tasks']}，对应 subject {protocol['affected_subjects']}；名义每用户 q={100.0 * protocol['nominal_proxy_fraction']:.0f}%，实际共替换 {protocol['proxy_sequences']}/{protocol['affected_task_sequences']} 条受影响用户 sequence（{100.0 * protocol['actual_proxy_fraction']:.2f}%），占完整上传流 {100.0 * protocol['stream_proxy_fraction']:.2f}%。保持 N→N、repeat=0、relative-L2≤{100.0 * protocol['max_relative_l2']:.0f}% 与 L∞/std≤{protocol['max_linf_over_std']:.2f}，使用同一 frozen manifest `{protocol['manifest_sha256']}`。",
        "",
        "| 方法 | Clean old ACC/MF1 | Shared old ACC/MF1 | Δ old ACC/MF1 | Clean new ACC/MF1 | Shared new ACC/MF1 | Δ new ACC/MF1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["methods"]:
        clean = row["clean"]
        shared = row["shared_proxy"]
        delta = row["delta"]
        lines.append(
            f"| {row['label']} | {fmt(clean['old_acc'])}/{fmt(clean['old_mf1'])} | "
            f"{fmt(shared['old_acc'])}/{fmt(shared['old_mf1'])} | "
            f"{delta_fmt(delta['old_acc'])}/{delta_fmt(delta['old_mf1'])} | "
            f"{fmt(clean['new_acc'])}/{fmt(clean['new_mf1'])} | "
            f"{fmt(shared['new_acc'])}/{fmt(shared['new_mf1'])} | "
            f"{delta_fmt(delta['new_acc'])}/{delta_fmt(delta['new_mf1'])} |"
        )
    lines.extend(
        [
            "",
            "负差值表示 shared proxy 噪声相对 clean 退化。本表复用 canonical clean 基线；两条 shared 轨迹在首个受扰动 task 前均通过逐字段一致性校验。结果仍是单 seed 点估计。",
        ]
    )
    return "\n".join(lines) + "\n"


def update_bci(path: Path, markdown: str) -> None:
    start = "<!-- FOCUSED_MULTI_USER_N2N_START -->"
    end = "<!-- FOCUSED_MULTI_USER_N2N_END -->"
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
    generation = load_json(root / "shared_payload" / "generation_summary.json")
    manifest = load_json(Path(generation["manifest"]))
    affected_tasks = [int(task) for task in generation["affected_tasks"]]
    generations = generation["generations"]
    first_affected = min(affected_tasks)

    rows = []
    for label, family, method in METHODS:
        clean_path = metrics_path(clean_root, family, method, "clean")
        shared_path = metrics_path(root, family, method, "shared_proxy")
        clean_metrics = load_json(clean_path)
        shared_metrics = load_json(shared_path)
        prefix = validate_paired_prefix(
            clean_metrics,
            shared_metrics,
            affected_task=first_affected,
            expected_tasks=49,
            label=label,
        )
        clean = metric_view(clean_metrics)
        shared = metric_view(shared_metrics)
        delta = {key: shared[key] - clean[key] for key in clean}
        rows.append(
            {
                "label": label,
                "family": family,
                "method": method,
                "clean_metrics": str(clean_path),
                "shared_metrics": str(shared_path),
                "clean": clean,
                "shared_proxy": shared,
                "delta": delta,
                "paired_prefix_validation": prefix,
            }
        )

    affected_task_sequences = sum(int(row["total_sequences"]) for row in generations)
    report = {
        "protocol": {
            "name": "focused multi-user frozen shared proxy N-to-N",
            "seed": int(manifest["split"]["seed"]),
            "affected_tasks": affected_tasks,
            "affected_subjects": generation["affected_subjects"],
            "proxy_sequences": int(generation["proxy_sequences"]),
            "affected_task_sequences": affected_task_sequences,
            "uploaded_sequences": int(generation["uploaded_sequences"]),
            "nominal_proxy_fraction": float(manifest["constraints"]["sequence_fraction"]),
            "actual_proxy_fraction": generation["proxy_sequences"] / affected_task_sequences,
            "stream_proxy_fraction": generation["proxy_sequences"] / generation["uploaded_sequences"],
            "max_relative_l2": float(manifest["constraints"]["max_relative_l2"]),
            "max_linf_over_std": float(manifest["constraints"]["max_linf_over_std"]),
            "repeat": int(manifest["constraints"]["repeat"]),
            "optimization_steps": sorted({int(row["steps"]) for row in generations}),
            "manifest": generation["manifest"],
            "manifest_sha256": generation["manifest_sha256"],
            "clean_root": str(clean_root),
        },
        "methods": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "FULL_RESULTS.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = render(report)
    (root / "FULL_RESULTS_ZH.md").write_text(markdown, encoding="utf-8")
    update_bci(args.bci.resolve(), markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
