"""Canonical partial-replacement N-to-N EEG upload manifests.

The resolver operates on signal paths only. It never accepts or opens target
annotation paths, so callers must keep evaluation labels on the clean side of
their data pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "brainuicl-canonical-n2n-v1"
VERIFY_MODES = {"none", "selected", "full"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_signal(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Signal payload is not an ndarray: {path}")
    if value.ndim != 3:
        raise ValueError(f"Signal payload must have shape [epochs,channels,time]: {path}")
    return value


def signal_metadata(path: Path) -> dict[str, Any]:
    value = _load_signal(path)
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "sha256": sha256_file(path),
    }


def _relative_l2(delta: np.ndarray, clean: np.ndarray) -> float:
    denominator = float(np.linalg.norm(clean.reshape(-1).astype(np.float64)))
    numerator = float(np.linalg.norm(delta.reshape(-1).astype(np.float64)))
    if denominator <= 1e-12:
        return 0.0 if numerator <= 1e-12 else math.inf
    return numerator / denominator


def _linf_over_std(delta: np.ndarray, clean: np.ndarray) -> float:
    denominator = float(clean.astype(np.float64).std())
    numerator = float(np.abs(delta.astype(np.float64)).max(initial=0.0))
    if denominator <= 1e-12:
        return 0.0 if numerator <= 1e-12 else math.inf
    return numerator / denominator


def signal_delta_metadata(clean_path: Path, proxy_path: Path) -> dict[str, float]:
    clean = _load_signal(clean_path).astype(np.float32, copy=False)
    proxy = _load_signal(proxy_path).astype(np.float32, copy=False)
    if clean.shape != proxy.shape:
        raise ValueError(
            f"Clean/proxy shape mismatch: {clean_path} {clean.shape} != "
            f"{proxy_path} {proxy.shape}"
        )
    delta = proxy - clean
    channels = clean.shape[1]
    eog_end = min(2, channels)
    eeg_start = min(2, channels)
    metadata = {
        "relative_l2": _relative_l2(delta, clean),
        "linf_over_std": _linf_over_std(delta, clean),
        "eog_relative_l2": _relative_l2(delta[:, :eog_end], clean[:, :eog_end]),
        "eog_linf_over_std": _linf_over_std(delta[:, :eog_end], clean[:, :eog_end]),
    }
    if eeg_start < channels:
        metadata.update(
            {
                "eeg_relative_l2": _relative_l2(
                    delta[:, eeg_start:], clean[:, eeg_start:]
                ),
                "eeg_linf_over_std": _linf_over_std(
                    delta[:, eeg_start:], clean[:, eeg_start:]
                ),
            }
        )
    else:
        metadata.update({"eeg_relative_l2": 0.0, "eeg_linf_over_std": 0.0})
    return metadata


def make_task_entry(
    *,
    task: int,
    subject: int,
    clean_paths: Sequence[Path],
    proxy_paths: Mapping[int, Path],
) -> dict[str, Any]:
    """Build and fully measure one ordered task entry."""

    invalid = sorted(set(proxy_paths) - set(range(len(clean_paths))))
    if invalid:
        raise ValueError(f"Proxy indices are outside the clean task: {invalid}")
    slots: list[dict[str, Any]] = []
    for slot, clean_value in enumerate(clean_paths):
        clean_path = Path(clean_value).resolve()
        clean_meta = signal_metadata(clean_path)
        is_proxy = slot in proxy_paths
        proxy_path = Path(proxy_paths[slot]).resolve() if is_proxy else None
        proxy_meta = signal_metadata(proxy_path) if proxy_path is not None else None
        if proxy_meta is not None:
            if proxy_meta["shape"] != clean_meta["shape"]:
                raise ValueError(f"Proxy shape differs at task {task}, slot {slot}")
            if proxy_meta["dtype"] != clean_meta["dtype"]:
                raise ValueError(f"Proxy dtype differs at task {task}, slot {slot}")
            delta = signal_delta_metadata(clean_path, proxy_path)
        else:
            delta = {
                "relative_l2": 0.0,
                "linf_over_std": 0.0,
                "eog_relative_l2": 0.0,
                "eog_linf_over_std": 0.0,
                "eeg_relative_l2": 0.0,
                "eeg_linf_over_std": 0.0,
            }
        slots.append(
            {
                "slot": int(slot),
                "sequence_index": int(clean_path.stem),
                "clean_path": str(clean_path),
                "clean_sha256": clean_meta["sha256"],
                "proxy_path": None if proxy_path is None else str(proxy_path),
                "proxy_sha256": None if proxy_meta is None else proxy_meta["sha256"],
                "is_proxy": bool(is_proxy),
                "shape": clean_meta["shape"],
                "dtype": clean_meta["dtype"],
                "clean_finite": clean_meta["finite"],
                "proxy_finite": None if proxy_meta is None else proxy_meta["finite"],
                **delta,
            }
        )
    proxy_count = sum(int(slot["is_proxy"]) for slot in slots)
    return {
        "task": int(task),
        "subject": int(subject),
        "uploaded_sequences": len(slots),
        "proxy_sequences": proxy_count,
        "proxy_fraction": proxy_count / max(len(slots), 1),
        "slots": slots,
    }


def write_manifest(
    path: Path,
    *,
    tasks: Sequence[dict[str, Any]],
    split: Mapping[str, Any],
    constraints: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "protocol": "partial content replacement with unchanged upload cardinality",
        "split": dict(split),
        "constraints": dict(constraints),
        "provenance": dict(provenance),
        "tasks": list(tasks),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"Unsupported N-to-N manifest schema {payload.get('schema')!r}; "
            f"expected {SCHEMA!r}"
        )
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Canonical N-to-N manifest must contain a non-empty task list")
    task_ids = [int(row.get("task", -1)) for row in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Canonical N-to-N manifest contains duplicate task IDs")
    return payload


def _task_row(payload: Mapping[str, Any], task: int) -> dict[str, Any]:
    for row in payload["tasks"]:
        if int(row.get("task", -1)) == int(task):
            return row
    raise KeyError(f"Canonical N-to-N manifest has no task {task}")


def _validate_slot_metadata(
    slot: Mapping[str, Any],
    clean_path: Path,
    selected_path: Path,
    *,
    is_proxy: bool,
    verify_clean_hash: bool,
) -> None:
    clean = _load_signal(clean_path)
    selected = _load_signal(selected_path)
    expected_shape = tuple(int(item) for item in slot["shape"])
    if clean.shape != expected_shape or selected.shape != expected_shape:
        raise ValueError(f"Manifest shape mismatch at slot {slot['slot']}")
    if str(clean.dtype) != slot["dtype"] or str(selected.dtype) != slot["dtype"]:
        raise ValueError(f"Manifest dtype mismatch at slot {slot['slot']}")
    if not np.isfinite(clean).all() or not np.isfinite(selected).all():
        raise ValueError(f"Non-finite canonical payload at slot {slot['slot']}")
    if verify_clean_hash and sha256_file(clean_path) != slot["clean_sha256"]:
        raise ValueError(f"Clean SHA-256 mismatch at slot {slot['slot']}")
    if is_proxy and sha256_file(selected_path) != slot["proxy_sha256"]:
        raise ValueError(f"Proxy SHA-256 mismatch at slot {slot['slot']}")
    if is_proxy:
        measured = signal_delta_metadata(clean_path, selected_path)
        for name, value in measured.items():
            expected = float(slot[name])
            if not math.isclose(value, expected, rel_tol=2e-5, abs_tol=2e-7):
                raise ValueError(
                    f"Manifest {name} mismatch at slot {slot['slot']}: "
                    f"measured={value}, recorded={expected}"
                )


def _validate_budget(slot: Mapping[str, Any], constraints: Mapping[str, Any]) -> None:
    if not bool(slot["is_proxy"]):
        return
    max_relative_l2 = constraints.get("max_relative_l2")
    max_linf_over_std = constraints.get("max_linf_over_std")
    tolerance = 1e-6
    if max_relative_l2 is not None:
        for name in ("relative_l2", "eog_relative_l2", "eeg_relative_l2"):
            if float(slot[name]) > float(max_relative_l2) + tolerance:
                raise ValueError(f"{name} exceeds canonical budget at slot {slot['slot']}")
    if max_linf_over_std is not None:
        for name in ("linf_over_std", "eog_linf_over_std", "eeg_linf_over_std"):
            if float(slot[name]) > float(max_linf_over_std) + tolerance:
                raise ValueError(f"{name} exceeds canonical budget at slot {slot['slot']}")


@dataclass(frozen=True)
class ResolvedN2NTask:
    task: int
    subject: int
    data_paths: tuple[Path, ...]
    proxy_indices: tuple[int, ...]
    diagnostics: dict[str, Any]


def resolve_task(
    manifest_path: Path,
    *,
    task: int,
    subject: int,
    clean_data_paths: Sequence[Path],
    verify: str = "selected",
) -> ResolvedN2NTask:
    """Resolve one mixed task without accepting target annotation paths."""

    if verify not in VERIFY_MODES:
        raise ValueError(f"verify must be one of {sorted(VERIFY_MODES)}")
    manifest_path = Path(manifest_path).resolve()
    payload = load_manifest(manifest_path)
    row = _task_row(payload, task)
    if int(row.get("subject", -1)) != int(subject):
        raise ValueError(
            f"Manifest subject mismatch for task {task}: "
            f"expected {subject}, found {row.get('subject')}"
        )
    slots = row.get("slots")
    if not isinstance(slots, list) or len(slots) != len(clean_data_paths):
        raise ValueError(f"Manifest upload count mismatch for task {task}")
    root = manifest_path.parent
    selected_paths: list[Path] = []
    proxy_indices: list[int] = []
    constraints = payload.get("constraints", {})
    for index, (slot, clean_value) in enumerate(zip(slots, clean_data_paths)):
        clean_path = Path(clean_value).resolve()
        if int(slot.get("slot", -1)) != index:
            raise ValueError(f"Manifest slot ordering mismatch for task {task}")
        if int(slot.get("sequence_index", -1)) != int(clean_path.stem):
            raise ValueError(f"Manifest sequence index mismatch for task {task}, slot {index}")
        recorded_clean = _resolve_path(slot["clean_path"], root).resolve()
        if recorded_clean != clean_path:
            raise ValueError(
                f"Manifest clean path mismatch for task {task}, slot {index}: "
                f"{recorded_clean} != {clean_path}"
            )
        is_proxy = bool(slot.get("is_proxy", False))
        if is_proxy:
            if not slot.get("proxy_path") or not slot.get("proxy_sha256"):
                raise ValueError(f"Proxy slot lacks payload metadata at task {task}, slot {index}")
            selected = _resolve_path(slot["proxy_path"], root).resolve()
            proxy_indices.append(index)
        else:
            if slot.get("proxy_path") is not None or slot.get("proxy_sha256") is not None:
                raise ValueError(f"Clean slot unexpectedly names a proxy at task {task}, slot {index}")
            selected = clean_path
        if not selected.is_file():
            raise FileNotFoundError(f"Canonical payload is missing: {selected}")
        _validate_budget(slot, constraints)
        should_measure = verify == "full" or (verify == "selected" and is_proxy)
        if should_measure:
            _validate_slot_metadata(
                slot,
                clean_path,
                selected,
                is_proxy=is_proxy,
                verify_clean_hash=True,
            )
        selected_paths.append(selected)
    declared_count = int(row.get("proxy_sequences", -1))
    if declared_count != len(proxy_indices):
        raise ValueError(f"Manifest proxy count mismatch for task {task}")
    diagnostics = {
        "mode": "canonical_n2n_shared_proxy",
        "task": int(task),
        "subject": int(subject),
        "uploaded_sequences": len(selected_paths),
        "proxy_sequences": len(proxy_indices),
        "proxy_fraction": len(proxy_indices) / max(len(selected_paths), 1),
        "proxy_indices": proxy_indices,
        "n_to_n": True,
        "repeat": 0,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "verify_mode": verify,
    }
    return ResolvedN2NTask(
        task=int(task),
        subject=int(subject),
        data_paths=tuple(selected_paths),
        proxy_indices=tuple(proxy_indices),
        diagnostics=diagnostics,
    )


def validate_manifest(
    manifest_path: Path,
    task_inputs: Mapping[int, tuple[int, Sequence[Path]]],
) -> dict[str, Any]:
    """Fully validate all clean and proxy arrays against an expected stream."""

    manifest_path = Path(manifest_path).resolve()
    payload = load_manifest(manifest_path)
    declared_tasks = {int(row["task"]) for row in payload["tasks"]}
    expected_tasks = {int(task) for task in task_inputs}
    if declared_tasks != expected_tasks:
        raise ValueError(
            f"Manifest task set mismatch: declared={sorted(declared_tasks)}, "
            f"expected={sorted(expected_tasks)}"
        )
    proxy_count = 0
    upload_count = 0
    proxy_tasks: list[int] = []
    for task in sorted(task_inputs):
        subject, clean_paths = task_inputs[task]
        resolved = resolve_task(
            manifest_path,
            task=task,
            subject=subject,
            clean_data_paths=clean_paths,
            verify="full",
        )
        upload_count += len(resolved.data_paths)
        proxy_count += len(resolved.proxy_indices)
        if resolved.proxy_indices:
            proxy_tasks.append(int(task))
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "tasks": len(task_inputs),
        "uploaded_sequences": upload_count,
        "proxy_sequences": proxy_count,
        "proxy_tasks": proxy_tasks,
        "n_to_n": True,
        "all_slots_validated": True,
    }


__all__ = [
    "ResolvedN2NTask",
    "SCHEMA",
    "load_manifest",
    "make_task_entry",
    "resolve_task",
    "sha256_file",
    "signal_delta_metadata",
    "signal_metadata",
    "validate_manifest",
    "write_manifest",
]
