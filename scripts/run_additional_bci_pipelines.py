#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bandpass(x: np.ndarray, fs: float, low: float, high: float, axis: int = -1) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=axis)


def loo_nearest_centroid(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    labels = np.asarray(labels)
    preds = []
    classes = np.unique(labels)
    for i in range(len(labels)):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[i] = False
        x_train = features[train_mask]
        y_train = labels[train_mask]
        x_test = features[i : i + 1]
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train_z = (x_train - mean) / std
        x_test_z = (x_test - mean) / std
        centroids = []
        centroid_labels = []
        for cls in classes:
            cls_mask = y_train == cls
            if not np.any(cls_mask):
                continue
            centroids.append(x_train_z[cls_mask].mean(axis=0))
            centroid_labels.append(cls)
        centroids = np.stack(centroids)
        dists = np.linalg.norm(centroids - x_test_z, axis=1)
        preds.append(centroid_labels[int(np.argmin(dists))])
    preds = np.asarray(preds)
    per_class_acc = {}
    for cls in classes:
        mask = labels == cls
        per_class_acc[str(cls)] = float(np.mean(preds[mask] == labels[mask]))
    summary = {
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": float(np.mean(list(per_class_acc.values()))),
        "per_class_accuracy": per_class_acc,
    }
    return preds, summary


def parse_edf_header(raw: bytes) -> dict:
    ns = int(raw[252:256].decode("ascii").strip())
    header_size = 256 + ns * 256
    n_records = int(raw[236:244].decode("ascii").strip())
    record_duration = float(raw[244:252].decode("ascii").strip())
    pos = 256
    labels = [raw[pos + i * 16 : pos + (i + 1) * 16].decode("latin1").strip() for i in range(ns)]
    pos += ns * 16
    transducer = [raw[pos + i * 80 : pos + (i + 1) * 80].decode("latin1").strip() for i in range(ns)]
    pos += ns * 80
    phys_dim = [raw[pos + i * 8 : pos + (i + 1) * 8].decode("latin1").strip() for i in range(ns)]
    pos += ns * 8
    phys_min = np.array([float(raw[pos + i * 8 : pos + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)])
    pos += ns * 8
    phys_max = np.array([float(raw[pos + i * 8 : pos + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)])
    pos += ns * 8
    dig_min = np.array([float(raw[pos + i * 8 : pos + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)])
    pos += ns * 8
    dig_max = np.array([float(raw[pos + i * 8 : pos + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)])
    pos += ns * 8
    prefilter = [raw[pos + i * 80 : pos + (i + 1) * 80].decode("latin1").strip() for i in range(ns)]
    pos += ns * 80
    samples_per_record = np.array([int(raw[pos + i * 8 : pos + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)])
    return {
        "ns": ns,
        "header_size": header_size,
        "n_records": n_records,
        "record_duration": record_duration,
        "labels": labels,
        "transducer": transducer,
        "phys_dim": phys_dim,
        "phys_min": phys_min,
        "phys_max": phys_max,
        "dig_min": dig_min,
        "dig_max": dig_max,
        "prefilter": prefilter,
        "samples_per_record": samples_per_record,
    }


def normalize_eeg_label(label: str) -> str:
    return label.replace(".", "").upper()


def read_physionet_edf(path: Path) -> tuple[np.ndarray, list[str], float, list[dict]]:
    raw = path.read_bytes()
    h = parse_edf_header(raw)
    ann_idx = next(i for i, label in enumerate(h["labels"]) if "Annotations" in label)
    eeg_indices = [i for i in range(h["ns"]) if i != ann_idx]
    fs = float(h["samples_per_record"][eeg_indices[0]] / h["record_duration"])
    record_bytes = int(np.sum(h["samples_per_record"]) * 2)
    data = np.zeros((len(eeg_indices), h["n_records"] * h["samples_per_record"][eeg_indices[0]]), dtype=np.float32)
    ann_texts = []
    for rec_idx in range(h["n_records"]):
        start = h["header_size"] + rec_idx * record_bytes
        offset = 0
        for sig_idx in range(h["ns"]):
            n = int(h["samples_per_record"][sig_idx])
            chunk = raw[start + offset : start + offset + n * 2]
            offset += n * 2
            if sig_idx == ann_idx:
                ann_texts.append(chunk.decode("latin1", errors="ignore"))
                continue
            if sig_idx in eeg_indices:
                out_idx = eeg_indices.index(sig_idx)
                digital = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
                scale = (h["phys_max"][sig_idx] - h["phys_min"][sig_idx]) / (h["dig_max"][sig_idx] - h["dig_min"][sig_idx])
                physical = (digital - h["dig_min"][sig_idx]) * scale + h["phys_min"][sig_idx]
                lo = rec_idx * n
                data[out_idx, lo : lo + n] = physical
    tal = re.compile(r"([+-]\d+(?:\.\d+)?)(?:\x15(\d+(?:\.\d+)?))?\x14([^\x00]*)")
    annotations = []
    for text in ann_texts:
        for match in tal.finditer(text):
            onset, duration, desc = match.groups()
            for annotation in [a for a in desc.split("\x14") if a]:
                annotations.append(
                    {
                        "onset_s": float(onset),
                        "duration_s": float(duration) if duration else np.nan,
                        "annotation": annotation,
                    }
                )
    labels = [h["labels"][i] for i in eeg_indices]
    return data, labels, fs, annotations


def run_mi_pipeline() -> None:
    out = PROCESSED / "mi_eeg_s001_pipeline"
    out.mkdir(parents=True, exist_ok=True)
    run_maps = {
        "S001R03.edf": {"task": "executed_left_right_fist", "T1": "left_fist", "T2": "right_fist"},
        "S001R04.edf": {"task": "imagined_left_right_fist", "T1": "left_fist", "T2": "right_fist"},
        "S001R06.edf": {"task": "imagined_both_fists_both_feet", "T1": "both_fists", "T2": "both_feet"},
    }
    motor_channels = ["FC3", "FC1", "FCZ", "FC2", "FC4", "C3", "C1", "CZ", "C2", "C4", "CP3", "CP1", "CPZ", "CP2", "CP4"]
    feature_bands = [(8, 12), (12, 16), (16, 24), (24, 30)]
    all_epochs = []
    all_features = []
    rows = []
    for filename, meta in run_maps.items():
        raw, channels, fs, annotations = read_physionet_edf(RAW / "physionet_eegmmi" / filename)
        lookup = {normalize_eeg_label(ch): idx for idx, ch in enumerate(channels)}
        selected_idx = [lookup[ch] for ch in motor_channels if ch in lookup]
        selected_names = [ch for ch in motor_channels if ch in lookup]
        selected = raw[selected_idx]
        filtered_8_30 = bandpass(selected, fs, 8, 30, axis=-1)
        for ann in annotations:
            code = ann["annotation"]
            if code not in {"T1", "T2"}:
                continue
            start = int(round((ann["onset_s"] + 0.5) * fs))
            stop = int(round((ann["onset_s"] + 4.0) * fs))
            if start < 0 or stop > filtered_8_30.shape[1]:
                continue
            epoch = filtered_8_30[:, start:stop]
            band_features = []
            for low, high in feature_bands:
                band_epoch = bandpass(epoch, fs, low, high, axis=-1)
                band_features.append(np.log(np.var(band_epoch, axis=-1) + 1e-8))
            features = np.concatenate(band_features)
            label = meta[code]
            all_epochs.append(epoch.astype(np.float32))
            all_features.append(features.astype(np.float32))
            rows.append(
                {
                    "dataset": "physionet_eegmmi",
                    "subject": "S001",
                    "run": filename,
                    "task": meta["task"],
                    "onset_s": ann["onset_s"],
                    "epoch_start_s": ann["onset_s"] + 0.5,
                    "epoch_stop_s": ann["onset_s"] + 4.0,
                    "annotation_code": code,
                    "label_name": label,
                }
            )
    epochs = np.stack(all_epochs)
    features = np.stack(all_features)
    labels = np.array([row["label_name"] for row in rows])
    label_table = pd.DataFrame(rows)
    label_table.to_csv(out / "labels.csv", index=False)
    np.savez_compressed(
        out / "step_mi_epochs_8_30hz.npz",
        x=epochs,
        labels=labels,
        fs=np.array(160.0, dtype=np.float32),
        channels=np.array(selected_names),
        epoch_window_s=np.array([0.5, 4.0], dtype=np.float32),
    )
    np.savez_compressed(
        out / "step_mi_logvar_features.npz",
        features=features,
        labels=labels,
        channels=np.array(selected_names),
        feature_bands=np.array(feature_bands, dtype=np.float32),
    )
    decode_rows = []
    for task in sorted(label_table["task"].unique()):
        idx = label_table["task"].to_numpy() == task
        preds, summary = loo_nearest_centroid(features[idx], labels[idx])
        task_rows = label_table[idx].copy()
        task_rows["predicted_label"] = preds
        task_rows["correct"] = task_rows["predicted_label"] == task_rows["label_name"]
        decode_rows.append(task_rows)
        save_json(out / f"decode_sanity_{task}.json", summary)
    pd.concat(decode_rows).to_csv(out / "decode_sanity_by_task.csv", index=False)
    save_json(
        out / "summary.json",
        {
            "method": "Motor EEG pipeline: 8-30 Hz preprocessing, task epochs, filter-bank log-variance features, nearest-centroid sanity decode.",
            "epochs_shape": list(epochs.shape),
            "features_shape": list(features.shape),
            "selected_channels": selected_names,
            "feature_bands_hz": feature_bands,
            "tasks": sorted(label_table["task"].unique().tolist()),
            "labels": sorted(np.unique(labels).tolist()),
        },
    )


def run_p300_pipeline() -> None:
    out = PROCESSED / "p300_sub01_pipeline"
    out.mkdir(parents=True, exist_ok=True)
    eeg_path = RAW / "openneuro_p300" / "sub-01_ses-01_task-cnos_run-4_eeg.eeg"
    fs = 256.0
    channel_names = ["FZ", "Cz", "P3", "PZ", "P4", "PO7", "PO8", "Oz", "Timestamp"]
    raw = np.fromfile(eeg_path, dtype="<f4").reshape(-1, 9)
    eeg = raw[:, :8].T
    eeg = bandpass(eeg, fs, 0.1, 30, axis=-1)
    events = pd.read_csv(PROCESSED / "manifests" / "openneuro_ds003190_sub01_ses01_cnos_run4_p300_events.csv")
    pre = int(round(0.2 * fs))
    post = int(round(0.8 * fs))
    epochs = []
    labels = []
    kept_rows = []
    for _, row in events.iterrows():
        center = int(row["sample"])
        start = center - pre
        stop = center + post
        if start < 0 or stop > eeg.shape[1]:
            continue
        epoch = eeg[:, start:stop].copy()
        baseline = epoch[:, :pre].mean(axis=1, keepdims=True)
        epoch -= baseline
        epochs.append(epoch.astype(np.float32))
        labels.append(row["label_name"])
        kept_rows.append(row.to_dict())
    epochs = np.stack(epochs)
    labels = np.array(labels)
    windows = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]
    feature_chunks = []
    for start_s, stop_s in windows:
        start = pre + int(round(start_s * fs))
        stop = pre + int(round(stop_s * fs))
        feature_chunks.append(epochs[:, :, start:stop].mean(axis=-1))
    features = np.concatenate(feature_chunks, axis=1)
    preds, summary = loo_nearest_centroid(features, labels)
    label_table = pd.DataFrame(kept_rows)
    label_table["predicted_label"] = preds
    label_table["correct"] = preds == labels
    label_table.to_csv(out / "decode_sanity.csv", index=False)
    np.savez_compressed(
        out / "step_p300_epochs_0p1_30hz.npz",
        x=epochs,
        labels=labels,
        fs=np.array(fs, dtype=np.float32),
        channels=np.array(channel_names[:8]),
        epoch_window_s=np.array([-0.2, 0.8], dtype=np.float32),
    )
    np.savez_compressed(
        out / "step_p300_window_mean_features.npz",
        features=features.astype(np.float32),
        labels=labels,
        channels=np.array(channel_names[:8]),
        windows_s=np.array(windows, dtype=np.float32),
    )
    save_json(out / "decode_sanity_summary.json", summary)
    save_json(
        out / "summary.json",
        {
            "method": "P300 pipeline: 0.1-30 Hz preprocessing, -0.2 to 0.8 s ERP epochs, baseline correction, 4 temporal window-mean features per channel.",
            "raw_shape_time_by_channel": list(raw.shape),
            "epochs_shape": list(epochs.shape),
            "features_shape": list(features.shape),
            "channels": channel_names[:8],
            "feature_windows_s": windows,
            "labels": sorted(np.unique(labels).tolist()),
        },
    )


def run_fnirs_pipeline() -> None:
    out = PROCESSED / "fnirs_subject01_pipeline"
    out.mkdir(parents=True, exist_ok=True)
    mat = sio.loadmat(RAW / "hybrid_eeg_fnirs" / "matdata" / "subject_01.mat", squeeze_me=True, struct_as_record=False)
    nfo = mat["nfo"]
    mrk = mat["mrk"]
    fs = float(nfo.fs)
    channel_names = [str(x) for x in np.ravel(nfo.clab)]
    signals = np.stack([np.asarray(mat[f"ch{i + 1}"]).ravel() for i in range(len(channel_names))], axis=0)
    filtered = bandpass(signals, fs, 0.01, 0.09, axis=-1)
    class_names = [str(x) for x in np.ravel(mrk.className)]
    y = np.asarray(mrk.y)
    event_times_s = np.asarray(mrk.time).ravel() / 1000.0
    epoch_start_s, epoch_stop_s = -5.0, 25.0
    n_epoch = int(round((epoch_stop_s - epoch_start_s) * fs))
    epochs = []
    labels = []
    rows = []
    for idx, onset_s in enumerate(event_times_s):
        start = int(round((onset_s + epoch_start_s) * fs))
        stop = start + n_epoch
        if start < 0 or stop > filtered.shape[1]:
            continue
        epoch = filtered[:, start:stop].copy()
        base_start = int(round((0.0 - epoch_start_s - 1.0) * fs))
        base_stop = int(round((0.0 - epoch_start_s) * fs))
        epoch -= epoch[:, base_start:base_stop].mean(axis=1, keepdims=True)
        label_idx = int(np.argmax(y[:, idx]))
        label = class_names[label_idx]
        epochs.append(epoch.astype(np.float32))
        labels.append(label)
        rows.append({"event_index": idx + 1, "onset_s": onset_s, "label_name": label})
    epochs = np.stack(epochs)
    labels = np.array(labels)
    windows = [(5.0, 10.0), (10.0, 15.0)]
    feature_chunks = []
    for start_s, stop_s in windows:
        start = int(round((start_s - epoch_start_s) * fs))
        stop = int(round((stop_s - epoch_start_s) * fs))
        feature_chunks.append(epochs[:, :, start:stop].mean(axis=-1))
    features = np.concatenate(feature_chunks, axis=1)
    preds, summary = loo_nearest_centroid(features, labels)
    label_table = pd.DataFrame(rows)
    label_table["predicted_label"] = preds
    label_table["correct"] = preds == labels
    label_table.to_csv(out / "decode_sanity.csv", index=False)
    np.savez_compressed(
        out / "step_fnirs_epochs_0p01_0p09hz.npz",
        x=epochs,
        labels=labels,
        fs=np.array(fs, dtype=np.float32),
        channels=np.array(channel_names),
        epoch_window_s=np.array([epoch_start_s, epoch_stop_s], dtype=np.float32),
    )
    np.savez_compressed(
        out / "step_fnirs_window_mean_features.npz",
        features=features.astype(np.float32),
        labels=labels,
        channels=np.array(channel_names),
        windows_s=np.array(windows, dtype=np.float32),
    )
    save_json(out / "decode_sanity_summary.json", summary)
    save_json(
        out / "summary.json",
        {
            "method": "fNIRS pipeline: 0.01-0.09 Hz hemodynamic filtering, -5 to 25 s epochs, -1 to 0 s baseline, mean HbO/HbR features at 5-10 and 10-15 s.",
            "raw_shape_channel_by_time": list(signals.shape),
            "epochs_shape": list(epochs.shape),
            "features_shape": list(features.shape),
            "channels": channel_names,
            "feature_windows_s": windows,
            "labels": sorted(np.unique(labels).tolist()),
        },
    )


def main() -> None:
    run_mi_pipeline()
    run_p300_pipeline()
    run_fnirs_pipeline()
    print("Wrote additional BCI pipelines under data/processed/")


if __name__ == "__main__":
    main()
