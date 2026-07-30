#!/usr/bin/env python3
"""Convert the NEMAR FACED BIDS release into subject-level EEG sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd


CLASS_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Sadness",
    "Neutral",
    "Amusement",
    "Inspiration",
    "Joy",
    "Tenderness",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def process_subject(
    subject_dir: Path,
    output_root: Path,
    *,
    sfreq: int,
    epoch_seconds: int,
    sequence_length: int,
) -> dict:
    subject = subject_dir.name
    eeg_dir = subject_dir / "eeg"
    bdf_path = next(eeg_dir.glob("*_eeg.bdf"))
    events_path = next(eeg_dir.glob("*_events.tsv"))
    events = pd.read_csv(events_path, sep="\t", encoding="utf-8-sig")
    events = events[events["binary_label"].isin(["positive", "negative", "neutral"])]

    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    raw.filter(0.5, 45.0, n_jobs=1, verbose="ERROR")
    raw.resample(sfreq, npad="auto", verbose="ERROR")

    epoch_samples = sfreq * epoch_seconds
    epochs: list[np.ndarray] = []
    labels: list[int] = []
    for row in events.itertuples(index=False):
        emotion = "Neutral" if row.binary_label == "neutral" else row.emotion_label
        label = CLASS_TO_ID[str(emotion)]
        count = int(float(row.duration) // epoch_seconds)
        for offset in range(count):
            start = int(round((float(row.onset) + offset * epoch_seconds) * sfreq))
            stop = start + epoch_samples
            if stop <= raw.n_times:
                epochs.append(raw.get_data(start=start, stop=stop).astype(np.float32))
                labels.append(label)

    if not epochs:
        raise RuntimeError(f"No usable epochs in {subject}")
    values = np.stack(epochs)
    targets = np.asarray(labels, dtype=np.int64)
    usable = (len(values) // sequence_length) * sequence_length
    values = values[:usable].reshape(-1, sequence_length, values.shape[1], epoch_samples)
    targets = targets[:usable].reshape(-1, sequence_length)

    # Normalize each channel within each epoch. This removes BDF unit-scale
    # differences while preserving temporal and spatial EEG structure.
    means = values.mean(axis=-1, keepdims=True)
    stds = values.std(axis=-1, keepdims=True)
    values = ((values - means) / np.maximum(stds, 1e-6)).astype(np.float32)

    target_dir = output_root / subject
    (target_dir / "data").mkdir(parents=True, exist_ok=True)
    (target_dir / "label").mkdir(parents=True, exist_ok=True)
    for index, (sequence, sequence_targets) in enumerate(zip(values, targets)):
        np.save(target_dir / "data" / f"{index}.npy", sequence)
        np.save(target_dir / "label" / f"{index}.npy", sequence_targets)
    return {
        "subject": subject,
        "clips": int(len(events)),
        "epochs": int(usable),
        "sequences": int(len(values)),
        "class_counts": np.bincount(targets.reshape(-1), minlength=9).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sfreq", type=int, default=250)
    parser.add_argument("--epoch-seconds", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--max-subjects", type=int, default=0)
    args = parser.parse_args()

    subjects = sorted(path for path in args.bids_root.glob("sub-*") if path.is_dir())
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, subject_dir in enumerate(subjects, 1):
        row = process_subject(
            subject_dir,
            args.output_root,
            sfreq=args.sfreq,
            epoch_seconds=args.epoch_seconds,
            sequence_length=args.sequence_length,
        )
        manifest.append(row)
        print(
            f"[{index:03d}/{len(subjects):03d}] {row['subject']}: "
            f"{row['epochs']} epochs, {row['sequences']} sequences",
            flush=True,
        )
    metadata = {
        "class_names": CLASS_NAMES,
        "sfreq": args.sfreq,
        "epoch_seconds": args.epoch_seconds,
        "sequence_length": args.sequence_length,
        "subjects": manifest,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
