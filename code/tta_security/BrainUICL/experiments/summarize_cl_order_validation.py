#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "final_old_acc",
    "final_old_mf1",
    "final_seen_acc",
    "final_seen_mf1",
    "old_aaa",
    "old_aaf1",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--random-clean-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.run_root / "runs/isruc").glob("*/*/seed*/summary.json")):
        order, method, seed_dir = path.parts[-4:-1]
        selected_document = json.loads(path.read_text())
        selected = selected_document.get(method, selected_document)
        random_document = json.loads((args.random_clean_root / f"isruc/{method}/{seed_dir}/summary.json").read_text())
        random = random_document.get(method, random_document)
        row = {"seed": int(seed_dir.removeprefix("seed")), "method": method, "order": order}
        for metric in METRICS:
            row[f"selected_{metric}"] = float(selected[metric])
            row[f"random_{metric}"] = float(random[metric])
            row[f"delta_{metric}_pp"] = 100 * (float(selected[metric]) - float(random[metric]))
        rows.append(row)
    (args.run_root / "RESULTS.json").write_text(json.dumps({"rows": rows}, indent=2))
    lines = [
        "# Cross-partition Validation of Selected CL Orders", "",
        "Deltas are selected order minus the same-partition seed-random clean run, in percentage points.", "",
        "| Seed | Method | Selected order | Final old ACC/MF1 delta | Final seen-new ACC/MF1 delta | AAA/AAF1 delta |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['method']} | {row['order']} | "
            f"{row['delta_final_old_acc_pp']:+.3f}/{row['delta_final_old_mf1_pp']:+.3f} | "
            f"{row['delta_final_seen_acc_pp']:+.3f}/{row['delta_final_seen_mf1_pp']:+.3f} | "
            f"{row['delta_old_aaa_pp']:+.3f}/{row['delta_old_aaf1_pp']:+.3f} |"
        )
    (args.run_root / "SUMMARY_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"runs": len(rows)}))


if __name__ == "__main__":
    main()
