#!/usr/bin/env python3
"""Summarize paired Clean/Shared-proxy full49 runs across CL families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = (
    ("finetune", "regularization", "Finetune"),
    ("ewc", "regularization", "EWC"),
    ("online_ewc", "regularization", "Online EWC"),
    ("si", "regularization", "SI"),
    ("mas", "regularization", "MAS"),
    ("plain_er", "replay", "Plain ER"),
    ("full_spr_eeg_adapted", "full_spr", "Full SPR-EEG adapted"),
    ("full_puridiver_eeg_adapted", "full_puridiver", "Full PuriDivER-EEG adapted"),
)
CONDITION_ONLY_TASK_KEYS = {
    "attack",
    "attack_label_guiding_cpc_losses",
    "noise",
    "proxy_upload",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def result_paths(root: Path, method: str, family: str, condition: str):
    if family == "regularization":
        base = root / "runs" / family / method / condition
        return base / method / "metrics.json", base / "summary.json"
    base = root / "runs" / family / condition
    return base / "metrics.json", base / "summary.json"


def metric_row(root: Path, method: str, family: str, condition: str):
    metrics_path, summary_path = result_paths(root, method, family, condition)
    metrics = load_json(metrics_path)
    summaries = load_json(summary_path)
    summary = summaries[method]
    old = metrics["final"]["old_generalization"]
    return {
        "old_acc": float(old["acc"]),
        "old_mf1": float(old["mf1"]),
        "new_acc": float(summary["final_seen_acc"]),
        "new_mf1": float(summary["final_seen_mf1"]),
        "bwt_acc": float(summary["bwt_acc"]),
        "bwt_mf1": float(summary["bwt_mf1"]),
        "metrics": str(metrics_path.resolve()),
    }, metrics


def validate_paired_prefix(
    clean: dict[str, Any],
    shared: dict[str, Any],
    *,
    affected_task: int,
    expected_tasks: int,
    label: str,
) -> dict[str, Any]:
    """Require paired trajectories to be identical before signal replacement."""

    expected_ids = list(range(1, expected_tasks + 1))
    for condition, metrics in (("clean", clean), ("shared_proxy", shared)):
        task_ids = [int(row["task"]) for row in metrics.get("tasks", [])]
        if task_ids != expected_ids:
            raise ValueError(
                f"{label} {condition} task IDs are not complete 1..{expected_tasks}: "
                f"{task_ids}"
            )
    if clean.get("initial") != shared.get("initial"):
        raise ValueError(f"{label} paired initial metrics differ")

    compared = 0
    for clean_row, shared_row in zip(clean["tasks"], shared["tasks"]):
        task = int(clean_row["task"])
        if task >= affected_task:
            break
        clean_behavior = {
            key: value
            for key, value in clean_row.items()
            if key not in CONDITION_ONLY_TASK_KEYS
        }
        shared_behavior = {
            key: value
            for key, value in shared_row.items()
            if key not in CONDITION_ONLY_TASK_KEYS
        }
        if clean_behavior != shared_behavior:
            raise ValueError(
                f"{label} paired trajectories diverge before affected task "
                f"at task {task}"
            )
        compared += 1
    if compared != affected_task - 1:
        raise ValueError(
            f"{label} compared {compared} prefix tasks; expected {affected_task - 1}"
        )
    return {
        "identical_before_affected_task": True,
        "tasks_compared": compared,
    }


def fmt(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def delta_fmt(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def render(report: dict[str, Any]) -> str:
    manifest = report["manifest"]
    lines = [
        "## Canonical N→N 单用户 Shared-proxy Full49 结果",
        "",
        f"固定条件：seed {manifest['seed']}，task {manifest['affected_task']} / subject {manifest['affected_subject']}，名义 q={100.0 * manifest['nominal_proxy_fraction']:.0f}%，实际替换 {manifest['proxy_sequences']}/{manifest['affected_task_sequences']} 条 sequence（{100.0 * manifest['actual_proxy_fraction']:.2f}%），上传数量与顺序不变，repeat=0，所有方法读取同一 manifest `{manifest['sha256']}`。负差值表示 Shared-proxy 相对 Clean 退化。",
        "",
        "| 方法 | Clean old ACC/MF1 | Shared old ACC/MF1 | Δ old ACC/MF1 | Clean new ACC/MF1 | Shared new ACC/MF1 | Δ new ACC/MF1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["methods"]:
        clean, shared, delta = row["clean"], row["shared_proxy"], row["delta"]
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
            "以上为 seed 4321 的单次严格配对点估计，用于比较当前固定 proxy 噪声在各 CL 方法上的相对响应；它不是多 seed 均值或统计显著性结论。各方法的算法内更新流程与计算量不同，因此只解释同一方法内的 Shared-proxy − Clean 差值，不用绝对 ACC/MF1 对方法作抗噪声排名。",
        ]
    )
    return "\n".join(lines) + "\n"


def update_bci(path: Path, markdown: str) -> None:
    start = "<!-- CANONICAL_N2N_FULL49_START -->"
    end = "<!-- CANONICAL_N2N_FULL49_END -->"
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
    parser.add_argument("--bci", type=Path, default=Path("/home/undefined/Desktop/IPhone/BCI.md"))
    args = parser.parse_args()
    root = args.run_root.resolve()
    generation = load_json(root / "shared_payload" / "generation_summary.json")
    manifest_payload = load_json(Path(generation["manifest"]))
    affected_task = int(generation["affected_task"])
    expected_tasks = int(generation["validation"]["tasks"])
    rows = []
    for method, family, label in METHODS:
        clean, clean_metrics = metric_row(root, method, family, "clean")
        shared, shared_metrics = metric_row(root, method, family, "shared_proxy")
        paired_prefix = validate_paired_prefix(
            clean_metrics,
            shared_metrics,
            affected_task=affected_task,
            expected_tasks=expected_tasks,
            label=label,
        )
        delta = {
            key: shared[key] - clean[key]
            for key in ("old_acc", "old_mf1", "new_acc", "new_mf1", "bwt_acc", "bwt_mf1")
        }
        rows.append(
            {
                "method": method,
                "family": family,
                "label": label,
                "clean": clean,
                "shared_proxy": shared,
                "delta": delta,
                "paired_prefix_validation": paired_prefix,
            }
        )
    report = {
        "protocol": "canonical partial N-to-N shared proxy full49",
        "manifest": {
            "path": generation["manifest"],
            "sha256": generation["manifest_sha256"],
            "affected_task": generation["affected_task"],
            "affected_subject": generation["affected_subject"],
            "proxy_sequences": generation["proxy_sequences"],
            "uploaded_sequences": generation["uploaded_sequences"],
            "affected_task_sequences": generation["generation"]["total_sequences"],
            "nominal_proxy_fraction": manifest_payload["constraints"]["sequence_fraction"],
            "actual_proxy_fraction": generation["generation"]["poison_fraction"],
            "seed": manifest_payload["split"]["seed"],
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
