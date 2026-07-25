#!/usr/bin/env python3
"""Derive nested dose manifests from one maximum frozen N-to-N payload."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from n2n_shared_proxy import (
    load_manifest,
    sha256_file,
    validate_manifest,
    write_manifest,
)


TASK_PRIORITY = (
    25,
    13,
    37,
    1,
    49,
    7,
    19,
    31,
    43,
    5,
    11,
    17,
    23,
    29,
    35,
    41,
    47,
    3,
    9,
    15,
    21,
    27,
    33,
    39,
    45,
)

STAGES = (
    ("k01_q20", 1, 0.20),
    ("k05_q20", 5, 0.20),
    ("k05_q50", 5, 0.50),
    ("k10_q50", 10, 0.50),
    ("k25_q50", 25, 0.50),
    ("k25_q100", 25, 1.00),
)

DELTA_FIELDS = (
    "relative_l2",
    "linf_over_std",
    "eog_relative_l2",
    "eog_linf_over_std",
    "eeg_relative_l2",
    "eeg_linf_over_std",
)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def nested_slot_indices(
    total: int,
    fraction: float,
    *,
    seed: int,
    task: int,
    subject: int,
) -> frozenset[int]:
    if total < 1 or not 0.0 < fraction <= 1.0:
        raise ValueError("Nested slot selection requires total>0 and fraction in (0,1]")
    count = min(int(math.ceil(total * fraction)), total)
    rng = np.random.default_rng(int(seed) + 1009 * int(task) + 9173 * int(subject))
    order = rng.permutation(total)
    return frozenset(int(index) for index in order[:count])


def _clean_slot(slot: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(slot)
    result.update(
        {
            "proxy_path": None,
            "proxy_sha256": None,
            "is_proxy": False,
            "proxy_finite": None,
        }
    )
    result.update({name: 0.0 for name in DELTA_FIELDS})
    return result


def derive_task_rows(
    maximum: dict[str, Any],
    *,
    selected_tasks: Iterable[int],
    fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    selected_task_set = {int(task) for task in selected_tasks}
    rows: list[dict[str, Any]] = []
    for source_row in maximum["tasks"]:
        row = copy.deepcopy(source_row)
        task = int(row["task"])
        subject = int(row["subject"])
        selected_slots = (
            nested_slot_indices(
                len(row["slots"]),
                fraction,
                seed=seed,
                task=task,
                subject=subject,
            )
            if task in selected_task_set
            else frozenset()
        )
        slots = []
        for index, slot in enumerate(row["slots"]):
            if index in selected_slots:
                if not bool(slot.get("is_proxy")):
                    raise ValueError(
                        f"Maximum manifest lacks proxy payload at task {task}, slot {index}"
                    )
                slots.append(copy.deepcopy(slot))
            else:
                slots.append(_clean_slot(slot))
        proxy_count = len(selected_slots)
        row.update(
            {
                "slots": slots,
                "proxy_sequences": proxy_count,
                "proxy_fraction": proxy_count / max(len(slots), 1),
            }
        )
        rows.append(row)
    return rows


def run(args) -> dict[str, Any]:
    maximum_path = args.max_manifest.resolve()
    maximum = load_manifest(maximum_path)
    maximum_hash = sha256_file(maximum_path)
    declared_max_tasks = {
        int(task) for task in maximum.get("constraints", {}).get("affected_tasks", [])
    }
    if declared_max_tasks != set(TASK_PRIORITY):
        raise ValueError(
            "Maximum manifest task set does not match the fixed nested priority: "
            f"declared={sorted(declared_max_tasks)}"
        )
    if not math.isclose(
        float(maximum["constraints"].get("sequence_fraction", 0.0)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Maximum manifest must contain 100% proxy coverage")
    if any(
        not bool(slot["is_proxy"])
        for row in maximum["tasks"]
        if int(row["task"]) in declared_max_tasks
        for slot in row["slots"]
    ):
        raise ValueError("Maximum manifest has a clean slot inside an affected task")

    expected = {
        int(row["task"]): (
            int(row["subject"]),
            [Path(slot["clean_path"]) for slot in row["slots"]],
        )
        for row in maximum["tasks"]
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage_reports = []
    previous_task_set: set[int] = set()
    previous_slots: dict[int, frozenset[int]] = {}
    selection_seed = int(maximum["constraints"]["selection_seed"])

    for name, task_count, fraction in STAGES:
        selected_tasks = set(TASK_PRIORITY[:task_count])
        if not previous_task_set.issubset(selected_tasks):
            raise RuntimeError(f"Stage {name} violates nested task selection")
        rows = derive_task_rows(
            maximum,
            selected_tasks=selected_tasks,
            fraction=fraction,
            seed=selection_seed,
        )
        current_slots = {
            int(row["task"]): frozenset(
                int(slot["slot"]) for slot in row["slots"] if slot["is_proxy"]
            )
            for row in rows
        }
        for task, prior in previous_slots.items():
            if task in selected_tasks and not prior.issubset(current_slots[task]):
                raise RuntimeError(f"Stage {name} violates nested slots at task {task}")

        stage_root = args.output_root / "stages" / name
        manifest_path = stage_root / "manifest.json"
        constraints = copy.deepcopy(maximum["constraints"])
        constraints.update(
            {
                "affected_tasks": sorted(selected_tasks),
                "sequence_fraction": float(fraction),
                "nested_stage": name,
                "nested_task_priority": list(TASK_PRIORITY),
                "parent_manifest_sha256": maximum_hash,
            }
        )
        payload = write_manifest(
            manifest_path,
            tasks=rows,
            split=maximum["split"],
            constraints=constraints,
            provenance={
                "generator": "nested subset of one maximum frozen payload",
                "maximum_manifest": str(maximum_path),
                "maximum_manifest_sha256": maximum_hash,
                "proxy_arrays_reused": True,
                "target_annotations_opened": False,
            },
        )
        validation = validate_manifest(manifest_path, expected)
        report = {
            "stage": name,
            "task_count": int(task_count),
            "nominal_sequence_fraction": float(fraction),
            "affected_tasks": sorted(selected_tasks),
            "affected_subjects": [
                int(maximum["split"]["new_order"][task - 1])
                for task in sorted(selected_tasks)
            ],
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "proxy_sequences": int(validation["proxy_sequences"]),
            "uploaded_sequences": int(validation["uploaded_sequences"]),
            "stream_proxy_fraction": validation["proxy_sequences"]
            / validation["uploaded_sequences"],
            "validation": validation,
            "schema": payload["schema"],
        }
        save_json(stage_root / "stage_summary.json", report)
        stage_reports.append(report)
        previous_task_set = selected_tasks
        previous_slots = current_slots

    sweep = {
        "schema": "brainuicl-nested-n2n-dose-sweep-v1",
        "maximum_manifest": str(maximum_path),
        "maximum_manifest_sha256": maximum_hash,
        "task_priority": list(TASK_PRIORITY),
        "stages": stage_reports,
    }
    save_json(args.output_root / "SWEEP_MANIFESTS.json", sweep)
    print(json.dumps(sweep, indent=2, ensure_ascii=False), flush=True)
    return sweep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    return args


if __name__ == "__main__":
    run(parse_args())
