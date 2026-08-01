#!/usr/bin/env python3
"""Summarize the paired PERSIST-EEG order matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("final_old_acc", "final_old_mf1", "final_seen_acc", "final_seen_mf1")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def method_summary(path: Path, method: str):
    return read_json(path)[method]


def metric_path(root: Path, dataset: str, schedule: str, method: str, seed: int):
    base = root / "runs" / dataset / schedule / method / f"seed{seed}"
    return base / "ewc" / "metrics.json" if method == "ewc" else base / "metrics.json"


def transcript_path(root: Path, dataset: str, schedule: str, method: str, seed: int):
    base = root / "runs" / dataset / schedule / method / f"seed{seed}"
    return (
        base / "ewc" / "progressive_proxy" / "transcript.json"
        if method == "ewc"
        else base / "progressive_proxy" / "transcript.json"
    )


def aggregate(rows):
    output = {}
    for metric in METRICS:
        values = [row["delta_pp"][metric] for row in rows]
        output[metric] = {
            "mean_pp": statistics.mean(values),
            "sample_std_pp": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_pp": min(values),
            "max_pp": max(values),
        }
    output["all_four_negative_runs"] = sum(
        all(row["delta_pp"][metric] < 0 for metric in METRICS) for row in rows
    )
    output["run_count"] = len(rows)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    datasets = ("isruc", "faced")
    methods = ("ewc", "plain_er")
    schedules = ("uniform_random", "stratified_random", "late_random")
    seeds = (4321, 4322, 4323)

    clean = {}
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                path = root / "clean" / dataset / method / f"seed{seed}" / "summary.json"
                clean[(dataset, method, seed)] = method_summary(path, method)

    runs = []
    audits = []
    grouped = defaultdict(list)
    for dataset in datasets:
        for schedule in schedules:
            for method in methods:
                for seed in seeds:
                    base = root / "runs" / dataset / schedule / method / f"seed{seed}"
                    summary = method_summary(base / "summary.json", method)
                    baseline = clean[(dataset, method, seed)]
                    delta = {
                        metric: 100.0 * (summary[metric] - baseline[metric])
                        for metric in METRICS
                    }
                    row = {
                        "dataset": dataset,
                        "schedule": schedule,
                        "method": method,
                        "seed": seed,
                        "delta_pp": delta,
                    }
                    if method == "plain_er":
                        row["memory_proxy_fraction"] = summary["final_memory_poisoned_fraction"]
                        row["replay_proxy_fraction"] = summary["replay_poisoned_fraction"]
                    runs.append(row)
                    grouped[(dataset, schedule, method)].append(row)

                    transcript = read_json(transcript_path(root, dataset, schedule, method, seed))
                    task_rows = [item for item in transcript["tasks"] if item.get("proxy_sequences", 0) > 0]
                    metrics = read_json(metric_path(root, dataset, schedule, method, seed))
                    audits.append(
                        {
                            "dataset": dataset,
                            "schedule": schedule,
                            "method": method,
                            "seed": seed,
                            "proxy_tasks": len(task_rows),
                            "proxy_sequences": sum(item["proxy_sequences"] for item in task_rows),
                            "unmodified_sequences": sum(item["unmodified_sequences"] for item in task_rows),
                            "max_cumulative_relative_l2": max(item["max_cumulative_relative_l2"] for item in task_rows),
                            "direction_bank_size": transcript["direction_bank"]["bank_size"],
                            "probability_observations": transcript["persist_state"]["observations"],
                            "victim_parameters_visible": transcript["protocol"]["victim_parameters_visible"],
                            "pretrain_seed": metrics["config"].get("pretrain_seed") or seed,
                        }
                    )

    aggregates = {
        f"{dataset}/{schedule}/{method}": aggregate(rows)
        for (dataset, schedule, method), rows in grouped.items()
    }
    overall = {
        f"{dataset}/{method}": aggregate(
            [row for row in runs if row["dataset"] == dataset and row["method"] == method]
        )
        for dataset in datasets
        for method in methods
    }
    replay = {}
    for dataset in datasets:
        for schedule in schedules:
            selected = [
                row for row in runs
                if row["dataset"] == dataset
                and row["schedule"] == schedule
                and row["method"] == "plain_er"
            ]
            replay[f"{dataset}/{schedule}"] = {
                "mean_memory_proxy_fraction": statistics.mean(row["memory_proxy_fraction"] for row in selected),
                "mean_replay_proxy_fraction": statistics.mean(row["replay_proxy_fraction"] for row in selected),
            }

    audit_summary = {
        "clean_runs": len(clean),
        "persist_runs": len(runs),
        "all_proxy_tasks_equal_5": all(row["proxy_tasks"] == 5 for row in audits),
        "all_sequences_modified": all(row["unmodified_sequences"] == 0 for row in audits),
        "all_cumulative_relative_l2_within_0_20": all(row["max_cumulative_relative_l2"] <= 0.200001 for row in audits),
        "all_direction_banks_full": all(row["direction_bank_size"] == 4 for row in audits),
        "victim_parameters_never_visible": all(not row["victim_parameters_visible"] for row in audits),
        "all_pretrain_seed_4321": all(row["pretrain_seed"] == 4321 for row in audits),
        "max_observed_cumulative_relative_l2": max(row["max_cumulative_relative_l2"] for row in audits),
    }
    payload = {
        "protocol": {
            "datasets": datasets,
            "methods": methods,
            "schedules": schedules,
            "seeds": seeds,
            "proxy_budget_tasks": 5,
            "pretrain_seed": 4321,
        },
        "audit": audit_summary,
        "aggregates": aggregates,
        "overall": overall,
        "replay": replay,
        "runs": runs,
        "run_audits": audits,
    }
    (root / "RESULTS.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PERSIST-EEG K=5 Order Matrix",
        "",
        "Values are mean +/- sample standard deviation across three paired seeds, in percentage points. Order: old ACC / old MF1 / seen-new ACC / seen-new MF1.",
        "",
        "| Dataset | Schedule | Method | Paired delta (pp) | All four negative |",
        "|---|---|---|---:|---:|",
    ]
    for key, item in aggregates.items():
        dataset, schedule, method = key.split("/")
        values = []
        for metric in METRICS:
            values.append(f"{item[metric]['mean_pp']:+.3f} +/- {item[metric]['sample_std_pp']:.3f}")
        lines.append(f"| {dataset.upper()} | {schedule} | {method} | `{' / '.join(values)}` | {item['all_four_negative_runs']}/3 |")
    lines.extend(["", f"Audit: `{json.dumps(audit_summary, ensure_ascii=False)}`", ""])
    (root / "SUMMARY_ZH.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
