#!/usr/bin/env python3
"""Collect SPR TensorBoard accuracies and compare them with paper tables."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_ROOTS = (
    Path("/home/undefined/Disk/ai-storage/SPR_checkpoints/quick_matrix"),
    Path("/home/undefined/Disk/ai-storage/SPR_checkpoints/medium_matrix"),
    Path("/home/undefined/Disk/ai-storage/SPR_checkpoints/large_matrix"),
    Path("/home/undefined/Disk/ai-storage/SPR_checkpoints/short_runs"),
)

PAPER_ACCURACY = {
    ("mnist", "sym", 20): 85.4,
    ("mnist", "sym", 40): 86.7,
    ("mnist", "sym", 60): 84.8,
    ("mnist", "asym", 20): 86.8,
    ("mnist", "asym", 40): 86.0,
    ("cifar10", "sym", 20): 43.9,
    ("cifar10", "sym", 40): 43.0,
    ("cifar10", "sym", 60): 40.0,
    ("cifar10", "asym", 20): 44.5,
    ("cifar10", "asym", 40): 43.9,
    ("cifar100", "rndsym", 20): 21.5,
    ("cifar100", "rndsym", 40): 21.1,
    ("cifar100", "rndsym", 60): 18.1,
    ("cifar100", "supsym", 20): 20.5,
    ("cifar100", "supsym", 40): 19.8,
    ("cifar100", "supsym", 60): 16.5,
    ("webvision", "real", 0): 40.0,
}

RUN_RE = re.compile(
    r"(?:(?P<suite>quick|medium|large|paper|short)_)?(?P<dataset>mnist|cifar10|cifar100|webvision)_"
    r"(?P<noise>sym|asym|rndsym|supsym|real)(?P<corr>\d+|real)?"
    r"(?:_e\d+)?(?:_seed(?P<seed>\d+))?"
)


def parse_run_name(name: str) -> dict[str, object]:
    match = RUN_RE.search(name)
    if not match:
        return {"run": name}
    data = match.groupdict()
    corr_text = data.get("corr")
    if data["noise"] == "real" or corr_text in (None, "real"):
        corr = 0
    else:
        corr = int(corr_text)
    return {
        "run": name,
        "suite": data.get("suite") or "",
        "dataset": data["dataset"],
        "noise": data["noise"],
        "corruption_percent": corr,
        "seed": int(data["seed"]) if data.get("seed") else "",
    }


def final_overall(event_file: Path) -> tuple[str, int, float] | None:
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    for tag in ea.Tags().get("scalars", []):
        if tag.startswith("accuracy/") and tag.endswith("/overall"):
            values = ea.Scalars(tag)
            if values:
                last = values[-1]
                return tag, last.step, float(last.value) * 100.0
    return None


def collect(roots: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for event_file in sorted(root.glob("**/events.out.tfevents*")):
            run_dir = event_file.parent
            if run_dir in seen:
                continue
            seen.add(run_dir)
            row = parse_run_name(run_dir.name)
            row["path"] = str(run_dir)
            result = final_overall(event_file)
            if result is None:
                row.update({"tag": "", "step": "", "accuracy": ""})
            else:
                tag, step, accuracy = result
                row.update({"tag": tag, "step": step, "accuracy": accuracy})
            key = (row.get("dataset"), row.get("noise"), row.get("corruption_percent"))
            paper = PAPER_ACCURACY.get(key)
            row["paper_accuracy"] = paper if paper is not None else ""
            row["delta_vs_paper"] = (row["accuracy"] - paper) if paper is not None and row["accuracy"] != "" else ""
            rows.append(row)
    return rows


def print_markdown(rows: list[dict[str, object]]) -> None:
    columns = ["run", "dataset", "noise", "corruption_percent", "seed", "accuracy", "paper_accuracy", "delta_vs_paper", "path"]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            values.append(str(value))
        print("| " + " | ".join(values) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    rows = collect(args.roots)
    rows.sort(key=lambda row: (str(row.get("dataset", "")), str(row.get("noise", "")), str(row.get("corruption_percent", "")), str(row.get("seed", "")), str(row.get("run", ""))))
    print_markdown(rows)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted({key for row in rows for key in row.keys()})
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
