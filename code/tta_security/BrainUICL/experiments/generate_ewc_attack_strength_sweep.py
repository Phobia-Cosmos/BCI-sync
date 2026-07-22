#!/usr/bin/env python3
"""Materialize an EWC attack-strength and coverage sweep.

The existing full49 frozen-proxy run already contains one raw, band-limited
proxy direction for 20% of every frequent task.  This script reuses those
immutable directions and changes only the final projection budget, attacked
task count, and within-task sequence count.  Reusing the directions makes the
comparisons paired: a stronger result cannot be attributed to a new proxy
optimization or a different sequence mask.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


DEFAULT_DATA_ROOT = Path(
    "/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32"
)
DEFAULT_SOURCE_ROOT = Path(
    "experiments/frozen_proxy_frequency_shift/full49_seed4321"
)
DEFAULT_OUTPUT_ROOT = Path(
    "experiments/ewc_attack_strength_sweep/frozen_proxy_F-S"
)

# The L-infinity cap is scaled with the L2 target.  The final 10% row is a
# strong but still bounded stress level; it avoids the 0.20-std saturation that
# made the earlier nominal 10% screening only reach about 6.3% in practice.
STRENGTH_LEVELS = (
    ("s005", 0.005, 0.02),
    ("s010", 0.010, 0.04),
    ("s025", 0.025, 0.10),
    ("s050", 0.050, 0.20),
    ("s100", 0.100, 0.40),
)
FREQUENT_TASKS = tuple(range(1, 50, 2))
TASK_COUNTS = (1, 3, 10, 25)
SEQUENCE_FRACTIONS = (0.05, 0.10, 0.20)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def replace_with_symlink(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def save_array(destination: Path, value: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    np.save(destination, value.astype(np.float32, copy=False))


def nested_subset(values: list[int] | tuple[int, ...], count: int) -> list[int]:
    if count < 1 or count > len(values):
        raise ValueError(f"count {count} is outside 1..{len(values)}")
    positions = np.rint(np.linspace(0, len(values) - 1, count)).astype(int)
    selected = [int(values[position]) for position in positions]
    if len(set(selected)) != count:
        raise RuntimeError("Nested task subset contains duplicates")
    return selected


def project_modality(
    raw: np.ndarray,
    base: np.ndarray,
    relative_l2: float,
    linf_std_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    raw_flat = raw.reshape(-1).astype(np.float64)
    base_flat = base.reshape(-1).astype(np.float64)
    raw_norm = float(np.linalg.norm(raw_flat))
    base_norm = max(float(np.linalg.norm(base_flat)), 1e-12)
    base_std = max(float(base_flat.std()), 1e-12)
    max_abs = max(float(np.max(np.abs(raw_flat))), 1e-12)
    l2_scale = relative_l2 * base_norm / raw_norm
    linf_scale = linf_std_scale * base_std / max_abs
    scale = min(l2_scale, linf_scale)
    delta = (raw.astype(np.float64) * scale).astype(np.float32)
    delta_flat = delta.reshape(-1).astype(np.float64)
    return delta, {
        "relative_l2": float(np.linalg.norm(delta_flat) / base_norm),
        "linf_over_std": float(np.max(np.abs(delta_flat)) / base_std),
    }


def aggregate(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "max": float(array.max())}


def make_metadata(
    *,
    task: int,
    subject: int,
    stream_name: str,
    level_name: str,
    relative_l2: float,
    linf_std_scale: float,
    sequence_fraction: float,
    attack_tasks: set[int],
    all_indices: list[int],
    active_indices: list[int],
    eog_l2: list[float],
    eeg_l2: list[float],
    eog_linf: list[float],
    eeg_linf: list[float],
) -> dict:
    return {
        "task": int(task),
        "subject": int(subject),
        "stream": stream_name,
        "frequency": "frequent",
        "shifted": True,
        "strength_level": level_name,
        "relative_l2_budget": float(relative_l2),
        "linf_std_scale": float(linf_std_scale),
        "sequence_fraction_within_attacked_task": float(sequence_fraction),
        "attack_tasks": sorted(int(item) for item in attack_tasks),
        "attacked": task in attack_tasks,
        "uploaded": int(len(all_indices)),
        "noisy_sequences": int(len(active_indices)),
        "noise_fraction": float(len(active_indices) / max(len(all_indices), 1)),
        "noise_indices": [int(item) for item in active_indices],
        "signs": {str(item): 1 for item in active_indices},
        "sign_positive": int(len(active_indices)),
        "sign_negative": 0,
        "eog_relative_l2": aggregate(eog_l2),
        "eeg_relative_l2": aggregate(eeg_l2),
        "eog_linf_over_std": aggregate(eog_linf),
        "eeg_linf_over_std": aggregate(eeg_linf),
        "source_directions": "full49_seed4321/directions",
    }


def materialize_condition(
    *,
    condition_root: Path,
    data_root: Path,
    source_root: Path,
    new_order: list[int],
    level_name: str,
    relative_l2: float,
    linf_std_scale: float,
    task_count: int,
    sequence_fraction: float,
    condition_name: str,
) -> dict:
    attack_tasks = set(nested_subset(FREQUENT_TASKS, task_count))
    task_metadata: list[dict] = []
    total_noisy = 0
    total_uploaded = 0

    for task, subject in enumerate(new_order, start=1):
        clean_data_root = data_root / str(subject) / "data"
        clean_paths = sorted(
            clean_data_root.glob("*.npy"), key=lambda path: int(path.stem)
        )
        if not clean_paths:
            raise FileNotFoundError(f"No clean sequences found: {clean_data_root}")
        all_indices = list(range(len(clean_paths)))
        source_metadata_path = (
            source_root
            / "rel_l2_0500"
            / "F-S"
            / f"individual_{task}"
            / "metadata.json"
        )
        if task not in attack_tasks:
            full_indices: list[int] = []
        else:
            if not source_metadata_path.is_file():
                raise FileNotFoundError(
                    f"No source mask for task {task}: {source_metadata_path}"
                )
            full_indices = [
                int(item) for item in read_json(source_metadata_path)["noise_indices"]
            ]
        active_count = (
            int(math.ceil(len(clean_paths) * sequence_fraction))
            if task in attack_tasks
            else 0
        )
        if active_count > len(full_indices):
            raise ValueError(
                f"Task {task} needs {active_count} directions for fraction "
                f"{sequence_fraction}, but only {len(full_indices)} are cached"
            )
        active_indices = full_indices[:active_count]
        direction_root = source_root / "directions" / f"individual_{task}"
        data_root_out = condition_root / f"individual_{task}" / "data"

        for index, source_path in enumerate(clean_paths):
            if index not in set(active_indices):
                replace_with_symlink(data_root_out / f"{index}.npy", source_path)

        eog_l2: list[float] = []
        eeg_l2: list[float] = []
        eog_linf: list[float] = []
        eeg_linf: list[float] = []
        for index in active_indices:
            direction_path = direction_root / f"{index}.npy"
            if not direction_path.is_file():
                raise FileNotFoundError(f"No cached direction: {direction_path}")
            base = np.load(clean_paths[index]).astype(np.float32)
            raw = np.load(direction_path).astype(np.float32)
            if base.shape != raw.shape or base.ndim != 3:
                raise ValueError(
                    f"Shape mismatch for task {task}, sequence {index}: "
                    f"base={base.shape}, direction={raw.shape}"
                )
            delta_eog, eog_metrics = project_modality(
                raw[:, :2, :],
                base[:, :2, :],
                relative_l2,
                linf_std_scale,
            )
            delta_eeg, eeg_metrics = project_modality(
                raw[:, 2:, :],
                base[:, 2:, :],
                relative_l2,
                linf_std_scale,
            )
            attacked = base.copy()
            attacked[:, :2, :] += delta_eog
            attacked[:, 2:, :] += delta_eeg
            save_array(data_root_out / f"{index}.npy", attacked)
            eog_l2.append(eog_metrics["relative_l2"])
            eeg_l2.append(eeg_metrics["relative_l2"])
            eog_linf.append(eog_metrics["linf_over_std"])
            eeg_linf.append(eeg_metrics["linf_over_std"])

        metadata = make_metadata(
            task=task,
            subject=subject,
            stream_name=condition_name,
            level_name=level_name,
            relative_l2=relative_l2,
            linf_std_scale=linf_std_scale,
            sequence_fraction=sequence_fraction,
            attack_tasks=attack_tasks,
            all_indices=all_indices,
            active_indices=active_indices,
            eog_l2=eog_l2,
            eeg_l2=eeg_l2,
            eog_linf=eog_linf,
            eeg_linf=eeg_linf,
        )
        write_json(condition_root / f"individual_{task}" / "metadata.json", metadata)
        task_metadata.append(metadata)
        total_noisy += len(active_indices)
        total_uploaded += len(all_indices)

    return {
        "condition": condition_name,
        "level": level_name,
        "relative_l2_budget": relative_l2,
        "linf_std_scale": linf_std_scale,
        "task_count": task_count,
        "attack_tasks": sorted(attack_tasks),
        "sequence_fraction": sequence_fraction,
        "noisy_sequences": total_noisy,
        "uploaded_sequences": total_uploaded,
        "stream_coverage": total_noisy / max(total_uploaded, 1),
        "task_metadata": task_metadata,
    }


def build_conditions() -> list[dict]:
    conditions: list[dict] = []
    seen: set[str] = set()

    def add(name: str, level: str, task_count: int, fraction: float) -> None:
        if name in seen:
            return
        seen.add(name)
        level_row = next(row for row in STRENGTH_LEVELS if row[0] == level)
        conditions.append(
            {
                "name": name,
                "level": level,
                "relative_l2": level_row[1],
                "linf_std_scale": level_row[2],
                "task_count": task_count,
                "sequence_fraction": fraction,
            }
        )

    # Primary question: how much per-sequence strength is needed when the
    # attack is frequent and systematic?
    for level, _relative_l2, _linf in STRENGTH_LEVELS:
        add(f"strength_{level}_k25_q20", level, 25, 0.20)

    # Separate task-frequency and within-task coverage effects at the 5% level.
    for task_count in TASK_COUNTS:
        add(f"subjects_s050_k{task_count:02d}_q20", "s050", task_count, 0.20)
    for fraction in SEQUENCE_FRACTIONS:
        fraction_name = int(round(fraction * 100))
        add(
            f"sequences_s050_k25_q{fraction_name:02d}",
            "s050",
            25,
            fraction,
        )
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = read_json(args.source_root / "manifest.json")
    if not source_manifest.get("proxy_parameters_unchanged", False):
        raise RuntimeError("Source frozen-proxy manifest is not immutable")
    new_order = [int(item) for item in source_manifest["config"]["split"]["new_order"]]
    if len(new_order) != 49:
        raise ValueError(f"Expected 49 new tasks, found {len(new_order)}")

    conditions = build_conditions()
    rows: list[dict] = []
    for condition in conditions:
        row = materialize_condition(
            condition_root=args.output_root / condition["name"],
            data_root=args.data_root,
            source_root=args.source_root,
            new_order=new_order,
            level_name=condition["level"],
            relative_l2=condition["relative_l2"],
            linf_std_scale=condition["linf_std_scale"],
            task_count=condition["task_count"],
            sequence_fraction=condition["sequence_fraction"],
            condition_name=condition["name"],
        )
        rows.append({
            key: value for key, value in row.items() if key != "task_metadata"
        })
        print(
            f"[sweep-stream] {condition['name']} "
            f"noisy={row['noisy_sequences']}/{row['uploaded_sequences']} "
            f"tasks={row['task_count']}",
            flush=True,
        )

    manifest = {
        "protocol": "EWC frozen-proxy F-S strength/coverage sweep",
        "source_manifest": str(args.source_root / "manifest.json"),
        "source_proxy_parameters_unchanged": True,
        "data_root": str(args.data_root),
        "new_order": new_order,
        "direction_source": str(args.source_root / "directions"),
        "strength_levels": [
            {
                "name": name,
                "relative_l2": relative_l2,
                "linf_std_scale": linf,
            }
            for name, relative_l2, linf in STRENGTH_LEVELS
        ],
        "conditions": rows,
        "notes": [
            "All conditions use the same cached directions and nested masks.",
            "The shifted F-S sign is +1 for every modified sequence.",
            "Sequence fractions are limited to 5%, 10%, and 20% because the source direction cache covers 20% per task.",
            "The 10% strength row is a bounded stress level, not an unbounded attack.",
        ],
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
