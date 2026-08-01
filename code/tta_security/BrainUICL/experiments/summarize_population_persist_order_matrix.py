#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("final_old_acc", "final_old_mf1", "final_seen_acc", "final_seen_mf1")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_path(root: Path, dataset: str, schedule: str | None, method: str, seed: int) -> Path:
    base = root / dataset
    if schedule is not None:
        base = base / schedule
    base = base / method / f"seed{seed}"
    return base / "ewc" / "metrics.json" if method == "ewc" else base / "metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    audits = []
    for path in sorted((args.run_root / "runs").glob("*/*/*/seed*/summary.json")):
        seed_dir = path.parent
        method = seed_dir.parent.name
        schedule = seed_dir.parent.parent.name
        dataset = seed_dir.parent.parent.parent.name
        seed = int(seed_dir.name.removeprefix("seed"))
        proxy_metrics_path = metric_path(args.run_root / "runs", dataset, schedule, method, seed)
        clean_metrics_path = metric_path(args.clean_root, dataset, None, method, seed)
        proxy_metrics = read_json(proxy_metrics_path)
        clean_metrics = read_json(clean_metrics_path)
        deltas = {
            name: 100.0 * (float(proxy_metrics["summary"][name]) - float(clean_metrics["summary"][name]))
            for name in METRICS
        }
        transcript_paths = list(seed_dir.rglob("progressive_proxy/transcript.json"))
        if len(transcript_paths) != 1:
            raise RuntimeError(f"Expected one transcript under {seed_dir}, got {transcript_paths}")
        transcript = read_json(transcript_paths[0])
        protocol = transcript["protocol"]
        proxy_tasks = [row for row in transcript["tasks"] if row.get("proxy_sequences", 0) > 0]
        class_count = len(proxy_tasks[0]["population_class_counts"])
        observed_classes = [
            sum(int(row["population_class_counts"][index]) for row in proxy_tasks)
            for index in range(class_count)
        ]
        audit = {
            "dataset": dataset,
            "schedule": schedule,
            "method": method,
            "seed": seed,
            "base_subject_is_none": protocol.get("base_subject") is None,
            "population_mode": protocol.get("population_mode") is True,
            "victim_parameters_hidden": protocol.get("victim_parameters_visible") is False,
            "proxy_task_count": len(proxy_tasks),
            "all_sequences_modified": all(int(row["unmodified_sequences"]) == 0 for row in proxy_tasks),
            "balanced_classes_per_task": all(
                max(int(value) for value in row["population_class_counts"])
                - min(int(value) for value in row["population_class_counts"]) <= 1
                for row in proxy_tasks
            ),
            "all_classes_observed_across_proxy_tasks": all(value > 0 for value in observed_classes),
            "minimum_population_subjects": min(int(row["population_subject_count"]) for row in proxy_tasks),
            "max_cumulative_relative_l2": max(float(row["max_uploaded_cumulative_relative_l2"]) for row in proxy_tasks),
            "population_invariants": protocol.get("population_invariants", []),
        }
        audits.append(audit)
        rows.append({
            "dataset": dataset,
            "schedule": schedule,
            "method": method,
            "seed": seed,
            "delta_pp": deltas,
            "all_four_negative": all(value < 0.0 for value in deltas.values()),
        })

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["schedule"], row["method"])].append(row)
    aggregates = []
    for key, group in sorted(grouped.items()):
        values = {name: np.asarray([row["delta_pp"][name] for row in group]) for name in METRICS}
        aggregates.append({
            "dataset": key[0],
            "schedule": key[1],
            "method": key[2],
            "runs": len(group),
            "mean_delta_pp": {name: float(value.mean()) for name, value in values.items()},
            "sample_std_pp": {name: float(value.std(ddof=1)) if len(value) > 1 else 0.0 for name, value in values.items()},
            "all_four_negative_runs": sum(row["all_four_negative"] for row in group),
        })
    audit_summary = {
        "runs": len(audits),
        "all_population_mode": all(row["population_mode"] for row in audits),
        "all_without_fixed_subject": all(row["base_subject_is_none"] for row in audits),
        "all_victim_parameters_hidden": all(row["victim_parameters_hidden"] for row in audits),
        "all_have_five_proxy_tasks": all(row["proxy_task_count"] == 5 for row in audits),
        "all_sequences_modified": all(row["all_sequences_modified"] for row in audits),
        "all_tasks_class_balanced": all(row["balanced_classes_per_task"] for row in audits),
        "all_classes_observed_across_proxy_tasks": all(row["all_classes_observed_across_proxy_tasks"] for row in audits),
        "minimum_population_subjects": min((row["minimum_population_subjects"] for row in audits), default=0),
        "max_cumulative_relative_l2": max((row["max_cumulative_relative_l2"] for row in audits), default=0.0),
    }
    payload = {"rows": rows, "aggregates": aggregates, "audits": audits, "audit_summary": audit_summary}
    (args.run_root / "RESULTS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Population-PERSIST EEG Order Matrix",
        "",
        "All values are paired against the existing clean run with the same dataset, method, seed and individual order. Values are percentage points in old ACC / old MF1 / seen-new ACC / seen-new MF1 order.",
        "",
        "| Dataset | Schedule | Method | Paired delta mean +/- sample std | All four negative |",
        "|---|---|---|---:|---:|",
    ]
    for row in aggregates:
        values = " / ".join(
            f"{row['mean_delta_pp'][name]:+.3f} +/- {row['sample_std_pp'][name]:.3f}"
            for name in METRICS
        )
        lines.append(
            f"| {row['dataset'].upper()} | {row['schedule']} | {row['method']} | `{values}` | "
            f"{row['all_four_negative_runs']}/{row['runs']} |"
        )
    lines.extend(("", f"Audit: `{json.dumps(audit_summary, ensure_ascii=False)}`", ""))
    (args.run_root / "SUMMARY_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
