#!/usr/bin/env python3
"""Rank fixed-partition CL subject orders without collapsing stability/plasticity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("final_old_acc", "final_old_mf1", "final_seen_acc", "final_seen_mf1")


def load_rows(run_root: Path, manifest_root: Path) -> list[dict]:
    rows = []
    for path in sorted((run_root / "runs").glob("*/*/*/seed*/summary.json")):
        relative = path.relative_to(run_root / "runs").parts
        dataset, order, method, seed_dir = relative[:4]
        summary_document = json.loads(path.read_text())
        summary = summary_document.get(method, summary_document)
        manifest = json.loads((manifest_root / dataset / seed_dir / f"{order}.json").read_text())
        row = {
            "dataset": dataset.upper(), "order": order, "method": method,
            "seed": int(seed_dir.removeprefix("seed")),
            "new_order": manifest["new_order"], **manifest["diagnostics"],
        }
        row.update({metric: float(summary[metric]) for metric in METRICS})
        row.update({
            "old_aaa": float(summary["old_aaa"]),
            "old_aaf1": float(summary["old_aaf1"]),
            "mean_current_after_acc": float(summary["mean_current_after_acc"]),
            "mean_current_after_mf1": float(summary["mean_current_after_mf1"]),
            "bwt_acc": float(summary["bwt_acc"]),
            "bwt_mf1": float(summary["bwt_mf1"]),
        })
        rows.append(row)
    for dataset in sorted({row["dataset"] for row in rows}):
        for method in sorted({row["method"] for row in rows if row["dataset"] == dataset}):
            group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            matrix = np.asarray([[row[metric] for metric in METRICS] for row in group])
            z = (matrix - matrix.mean(axis=0)) / np.maximum(matrix.std(axis=0), 1e-12)
            for index, row in enumerate(group):
                row["balanced_score"] = float(z[index].mean())
                row["pareto_optimal"] = not any(
                    np.all(matrix[other] >= matrix[index]) and np.any(matrix[other] > matrix[index])
                    for other in range(len(group)) if other != index
                )
            ranking = sorted(group, key=lambda row: row["balanced_score"], reverse=True)
            for rank, row in enumerate(ranking, start=1):
                row["balanced_rank"] = rank
    return rows


def write_summary(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Fixed-partition CL Individual-order Study", "",
        "All orders use the same partition and initial checkpoint. Balanced rank is the mean within-group z-score of final old ACC/MF1 and final seen-new ACC/MF1; Pareto status is reported separately.", "",
        "| Dataset | Method | Rank | Order | Old ACC/MF1 | Seen-new ACC/MF1 | AAA/AAF1 | Pareto | Path length |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["dataset"], item["method"], item["balanced_rank"])):
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['balanced_rank']} | {row['order']} | "
            f"{row['final_old_acc']:.3f}/{row['final_old_mf1']:.3f} | "
            f"{row['final_seen_acc']:.3f}/{row['final_seen_mf1']:.3f} | "
            f"{row['old_aaa']:.3f}/{row['old_aaf1']:.3f} | "
            f"{'yes' if row['pareto_optimal'] else 'no'} | {row['path_length']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.run_root, args.manifest_root)
    result = {"rows": rows, "runs": len(rows), "metrics": list(METRICS)}
    (args.run_root / "RESULTS.json").write_text(json.dumps(result, indent=2))
    write_summary(args.run_root / "SUMMARY_ZH.md", rows)
    print(json.dumps({"runs": len(rows)}))


if __name__ == "__main__":
    main()
