#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "manifests"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_ssvep_manifest() -> None:
    freq_phase = sio.loadmat(RAW / "tsinghua_ssvep" / "Freq_Phase.mat")
    freqs = freq_phase["freqs"].ravel()
    phases = freq_phase["phases"].ravel()
    shape = sio.whosmat(RAW / "tsinghua_ssvep" / "S1.mat")[0][1]
    rows = []
    for target_idx in range(40):
        for block_idx in range(6):
            rows.append(
                {
                    "dataset": "tsinghua_ssvep_benchmark",
                    "subject": "S1",
                    "block": block_idx + 1,
                    "target_index": target_idx + 1,
                    "frequency_hz": float(freqs[target_idx]),
                    "phase_rad": float(phases[target_idx]),
                    "array_shape": "x".join(map(str, shape)),
                    "ground_truth": f"target_{target_idx + 1}",
                }
            )
    write_csv(OUT / "tsinghua_ssvep_S1_trials.csv", rows)


def create_nirs_manifest() -> None:
    mat = sio.loadmat(
        RAW / "hybrid_eeg_fnirs" / "matdata" / "subject_01.mat",
        squeeze_me=True,
        struct_as_record=False,
    )
    mrk = mat["mrk"]
    nfo = mat["nfo"]
    class_names = [str(x) for x in np.ravel(mrk.className)]
    y = np.asarray(mrk.y)
    times_ms = np.asarray(mrk.time).ravel()
    rows = []
    for idx, time_ms in enumerate(times_ms):
        label_idx = int(np.argmax(y[:, idx]))
        rows.append(
            {
                "dataset": "shin_hybrid_eeg_nirs",
                "subject": "subject_01",
                "event_index": idx + 1,
                "time_ms": int(time_ms),
                "onset_s": float(time_ms) / 1000.0,
                "sampling_frequency_hz": float(nfo.fs),
                "label_index": label_idx,
                "label_name": class_names[label_idx],
                "ground_truth": class_names[label_idx],
            }
        )
    write_csv(OUT / "shin_hybrid_eeg_nirs_subject_01_events.csv", rows)


def create_p300_manifest() -> None:
    events = pd.read_csv(
        RAW / "openneuro_p300" / "sub-01_ses-01_task-cnos_run-4_events.tsv",
        sep="\t",
    )
    rows = []
    block = 0
    target_symbol = None
    for _, row in events.iterrows():
        value = int(row["value"])
        if value == 201:
            block += 1
        if 101 <= value <= 109:
            target_symbol = value - 100
            continue
        if value in {200, 201, 202, 203}:
            continue
        if 1 <= value <= 9 and target_symbol is not None:
            is_target = int(value == target_symbol)
            rows.append(
                {
                    "dataset": "openneuro_ds003190_p300",
                    "subject": "sub-01",
                    "session": "ses-01",
                    "run": "task-cnos_run-4",
                    "block": block,
                    "onset_s": float(row["onset"]),
                    "sample": int(row["sample"]),
                    "stimulus_symbol": value,
                    "target_symbol": target_symbol,
                    "label_index": is_target,
                    "label_name": "target" if is_target else "non_target",
                    "ground_truth": "P300_target" if is_target else "P300_non_target",
                }
            )
    write_csv(OUT / "openneuro_ds003190_sub01_ses01_cnos_run4_p300_events.csv", rows)


def edf_signal_metadata(raw: bytes) -> tuple[int, int, list[str], list[int], int, float]:
    ns = int(raw[252:256].decode("ascii").strip())
    header_size = 256 + ns * 256
    n_records = int(raw[236:244].decode("ascii").strip())
    record_duration = float(raw[244:252].decode("ascii").strip())
    labels_start = 256
    labels = [
        raw[labels_start + i * 16 : labels_start + (i + 1) * 16]
        .decode("latin1")
        .strip()
        for i in range(ns)
    ]
    samples_start = 256 + ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    samples_per_record = [
        int(raw[samples_start + i * 8 : samples_start + (i + 1) * 8].decode("ascii").strip())
        for i in range(ns)
    ]
    return ns, header_size, labels, samples_per_record, n_records, record_duration


def parse_edf_annotations(path: Path) -> list[dict]:
    raw = path.read_bytes()
    ns, header_size, labels, samples_per_record, n_records, _ = edf_signal_metadata(raw)
    ann_idx = next(i for i, label in enumerate(labels) if "Annotations" in label)
    record_bytes = sum(samples_per_record) * 2
    ann_offset_in_record = sum(samples_per_record[:ann_idx]) * 2
    ann_bytes = samples_per_record[ann_idx] * 2
    rows = []
    tal = re.compile(r"([+-]\d+(?:\.\d+)?)(?:\x15(\d+(?:\.\d+)?))?\x14([^\x00]*)")
    for record_idx in range(n_records):
        start = header_size + record_idx * record_bytes + ann_offset_in_record
        text = raw[start : start + ann_bytes].decode("latin1", errors="ignore")
        for match in tal.finditer(text):
            onset, duration, desc = match.groups()
            annotations = [a for a in desc.split("\x14") if a]
            for annotation in annotations:
                rows.append(
                    {
                        "onset_s": float(onset),
                        "duration_s": float(duration) if duration else "",
                        "annotation": annotation,
                    }
                )
    return rows


def create_physionet_mi_manifest() -> None:
    run_maps = {
        "S001R03.edf": {"task": "executed_left_right_fist", "T1": "left_fist", "T2": "right_fist"},
        "S001R04.edf": {"task": "imagined_left_right_fist", "T1": "left_fist", "T2": "right_fist"},
        "S001R06.edf": {"task": "imagined_both_fists_both_feet", "T1": "both_fists", "T2": "both_feet"},
    }
    rows = []
    for filename, meta in run_maps.items():
        for ann in parse_edf_annotations(RAW / "physionet_eegmmi" / filename):
            code = ann["annotation"]
            label = {"T0": "rest", "T1": meta["T1"], "T2": meta["T2"]}.get(code, code)
            rows.append(
                {
                    "dataset": "physionet_eegmmi",
                    "subject": "S001",
                    "run": filename,
                    "task": meta["task"],
                    "onset_s": ann["onset_s"],
                    "duration_s": ann["duration_s"],
                    "annotation_code": code,
                    "label_name": label,
                    "ground_truth": label,
                }
            )
    write_csv(OUT / "physionet_eegmmi_S001_annotations.csv", rows)


def main() -> None:
    create_ssvep_manifest()
    create_nirs_manifest()
    create_p300_manifest()
    create_physionet_mi_manifest()
    print(f"Wrote manifests under {OUT}")


if __name__ == "__main__":
    main()
