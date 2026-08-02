#!/usr/bin/env python3
"""Pair invariant-preserving Population Proxy runs with same-order clean CL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("final_old_acc", "final_old_mf1", "final_seen_acc", "final_seen_mf1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    args = parser.parse_args()
    rows, audits = [], []
    for proxy_path in sorted((args.run_root / "runs/isruc").glob("*/ewc/seed4321/summary.json")):
        order = proxy_path.parts[-4]
        clean_path = args.clean_root / f"runs/isruc/{order}/ewc/seed4321/summary.json"
        proxy_document, clean_document = json.loads(proxy_path.read_text()), json.loads(clean_path.read_text())
        proxy = proxy_document.get("ewc", proxy_document)
        clean = clean_document.get("ewc", clean_document)
        transcript_path = proxy_path.parent / "ewc/progressive_proxy/transcript.json"
        transcript = json.loads(transcript_path.read_text())
        proxy_tasks = [row for row in transcript["tasks"] if row.get("kind") == "progressive_proxy"]
        row = {"order": order}
        for metric in METRICS:
            row[f"clean_{metric}"] = float(clean[metric])
            row[f"proxy_{metric}"] = float(proxy[metric])
            row[f"delta_{metric}_pp"] = 100.0 * (float(proxy[metric]) - float(clean[metric]))
        row["all_four_negative"] = all(row[f"delta_{metric}_pp"] < 0 for metric in METRICS)
        rows.append(row)
        audits.append({
            "order": order,
            "proxy_tasks": len(proxy_tasks),
            "all_sequences_modified": all(task["unmodified_sequences"] == 0 for task in proxy_tasks),
            "max_invariant_drift": max(task["max_invariant_drift"] for task in proxy_tasks),
            "max_step_relative_l2": max(task["max_step_relative_l2"] for task in proxy_tasks),
            "max_cumulative_relative_l2": max(task["max_cumulative_relative_l2"] for task in proxy_tasks),
            "victim_parameters_visible": transcript["protocol"]["victim_parameters_visible"],
        })
    audit = {
        "runs": len(rows),
        "all_have_five_proxy_tasks": all(item["proxy_tasks"] == 5 for item in audits),
        "all_sequences_modified": all(item["all_sequences_modified"] for item in audits),
        "max_invariant_drift": max(item["max_invariant_drift"] for item in audits),
        "max_step_relative_l2": max(item["max_step_relative_l2"] for item in audits),
        "max_cumulative_relative_l2": max(item["max_cumulative_relative_l2"] for item in audits),
        "all_victim_parameters_hidden": not any(item["victim_parameters_visible"] for item in audits),
    }
    (args.run_root / "RESULTS.json").write_text(json.dumps({"rows": rows, "audits": audits, "audit": audit}, indent=2))
    lines = [
        "# Invariant-preserving Population Proxy Across Fixed Orders", "",
        "Deltas are Proxy minus the paired clean run with the identical partition and subject order, in percentage points.", "",
        "| Order | Old ACC/MF1 delta | Seen-new ACC/MF1 delta | All four negative |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['order']} | {row['delta_final_old_acc_pp']:+.3f}/{row['delta_final_old_mf1_pp']:+.3f} | "
            f"{row['delta_final_seen_acc_pp']:+.3f}/{row['delta_final_seen_mf1_pp']:+.3f} | "
            f"{'yes' if row['all_four_negative'] else 'no'} |"
        )
    lines.extend(["", f"Audit: `{json.dumps(audit)}`"])
    (args.run_root / "SUMMARY_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
