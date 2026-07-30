#!/usr/bin/env python3
"""Build deterministic interleaved Proxy schedules for position experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_TASKS = {"ISRUC": 49, "FACED": 61}
PLACEMENTS = ("front", "middle", "tail")
STRENGTHS = (10, 20)


def proxy_tasks(total_tasks: int, strength: int, placement: str) -> list[int]:
    if placement not in PLACEMENTS:
        raise ValueError(f"Unknown placement: {placement}")
    if strength < 1:
        raise ValueError("Proxy strength must be positive")
    span = 2 * strength - 1
    if span > total_tasks:
        raise ValueError(
            f"Cannot interleave {strength} Proxy tasks in {total_tasks} tasks"
        )
    if placement == "front":
        start = 1
    elif placement == "middle":
        start = (total_tasks - span) // 2 + 1
    else:
        start = total_tasks - span + 1
    tasks = list(range(start, start + span, 2))
    if len(tasks) != strength or tasks[-1] > total_tasks:
        raise AssertionError("Generated Proxy schedule has invalid cardinality")
    return tasks


def clean_tasks(total_tasks: int, selected: list[int]) -> list[int]:
    selected_set = set(selected)
    return [task for task in range(1, total_tasks + 1) if task not in selected_set]


def schedule(dataset: str, strength: int, placement: str) -> dict:
    total_tasks = DATASET_TASKS[dataset]
    selected = proxy_tasks(total_tasks, strength, placement)
    return {
        "dataset": dataset,
        "total_tasks": total_tasks,
        "strength_proxy_tasks": strength,
        "placement": placement,
        "window_start": selected[0],
        "window_end": selected[-1],
        "proxy_tasks": selected,
        "clean_feedback_tasks": clean_tasks(total_tasks, selected),
    }


def manifest() -> dict:
    return {
        "schedule_definition": (
            "A 2K-1 task window is anchored at the front, center, or tail; "
            "Proxy and natural-clean tasks alternate inside the window."
        ),
        "datasets": {
            dataset: {
                f"k{strength}_{placement}": schedule(
                    dataset, strength, placement
                )
                for strength in STRENGTHS
                for placement in PLACEMENTS
            }
            for dataset in DATASET_TASKS
        },
    }


def comma_separated(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_TASKS))
    parser.add_argument("--strength", type=int, choices=STRENGTHS)
    parser.add_argument("--placement", choices=PLACEMENTS)
    parser.add_argument(
        "--field",
        choices=("proxy", "clean", "json", "manifest"),
        default="json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.field == "manifest":
        payload = manifest()
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        if args.dataset is None or args.strength is None or args.placement is None:
            raise SystemExit(
                "--dataset, --strength, and --placement are required"
            )
        payload = schedule(args.dataset, args.strength, args.placement)
        if args.field == "proxy":
            text = comma_separated(payload["proxy_tasks"])
        elif args.field == "clean":
            text = comma_separated(payload["clean_feedback_tasks"])
        else:
            text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
