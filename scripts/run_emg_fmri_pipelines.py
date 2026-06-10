#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bandpass(x: np.ndarray, fs: float, low: float, high: float, axis: int = -1) -> np.ndarray:
    high = min(high, fs * 0.49)
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=axis)


def nearest_centroid_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    y_train = np.asarray(y_train)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_z = (x_train - mean) / std
    x_test_z = (x_test - mean) / std
    centroids = []
    labels = []
    for label in np.unique(y_train):
        mask = y_train == label
        centroids.append(x_train_z[mask].mean(axis=0))
        labels.append(label)
    centroids = np.stack(centroids)
    dists = np.linalg.norm(x_test_z[:, None, :] - centroids[None, :, :], axis=2)
    return np.asarray(labels)[np.argmin(dists, axis=1)]


def loo_nearest_centroid(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    labels = np.asarray(labels)
    preds = []
    for i in range(len(labels)):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[i] = False
        preds.append(nearest_centroid_predict(features[train_mask], labels[train_mask], features[i : i + 1])[0])
    preds = np.asarray(preds)
    return preds, classification_summary(labels, preds)


def classification_summary(labels: np.ndarray, preds: np.ndarray) -> dict:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    per_class = {}
    for label in np.unique(labels):
        mask = labels == label
        per_class[str(label)] = float(np.mean(preds[mask] == labels[mask]))
    return {
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "per_class_accuracy": per_class,
    }


EMG_CLASS_NAMES = {
    0: "unmarked",
    1: "hand_at_rest",
    2: "fist",
    3: "wrist_flexion",
    4: "wrist_extension",
    5: "radial_deviation",
    6: "ulnar_deviation",
    7: "extended_palm",
}


def contiguous_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []
    segments = []
    start = 0
    current = int(labels[0])
    for idx in range(1, len(labels)):
        label = int(labels[idx])
        if label != current:
            segments.append((start, idx, current))
            start = idx
            current = label
    segments.append((start, len(labels), current))
    return segments


def emg_feature_names(channels: list[str]) -> np.ndarray:
    metrics = ["rms", "mav", "waveform_length", "zero_crossings", "slope_sign_changes"]
    return np.asarray([f"{ch}_{metric}" for metric in metrics for ch in channels])


def extract_emg_features(windows: np.ndarray) -> np.ndarray:
    # windows shape: n_windows, n_channels, n_samples
    diff = np.diff(windows, axis=2)
    threshold = 1e-5
    rms = np.sqrt(np.mean(windows**2, axis=2))
    mav = np.mean(np.abs(windows), axis=2)
    waveform_length = np.sum(np.abs(diff), axis=2)
    zero_crossings = np.sum(
        (windows[:, :, :-1] * windows[:, :, 1:] < 0) & (np.abs(diff) > threshold),
        axis=2,
    )
    slope_sign_changes = np.sum(
        (diff[:, :, :-1] * diff[:, :, 1:] < 0) & (np.abs(diff[:, :, 1:] - diff[:, :, :-1]) > threshold),
        axis=2,
    )
    return np.concatenate(
        [
            rms,
            mav,
            waveform_length,
            zero_crossings.astype(np.float32),
            slope_sign_changes.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def run_emg_pipeline() -> None:
    out = PROCESSED / "emg_gestures_pipeline"
    out.mkdir(parents=True, exist_ok=True)
    subject_dir = RAW / "emg_gestures" / "EMG_data_for_gestures-master" / "01"
    files = sorted(subject_dir.glob("*_raw_data_*.txt"))
    channels = [f"channel{i}" for i in range(1, 9)]

    raw_parts = []
    filtered_parts = []
    time_parts = []
    class_parts = []
    file_parts = []
    file_names = []
    per_file = []
    fs_values = []

    windows = []
    rows = []
    win_ms = 200
    stride_ms = 100

    for file_idx, path in enumerate(files):
        df = pd.read_csv(path, sep="\t")
        file_names.append(path.name)
        time_ms = df["time"].to_numpy(dtype=np.int64)
        x_raw = df[channels].to_numpy(dtype=np.float32)
        sample_labels = df["class"].to_numpy(dtype=np.int16)
        median_dt_ms = float(np.median(np.diff(time_ms)))
        fs = 1000.0 / median_dt_ms
        fs_values.append(fs)
        x_filtered = bandpass(x_raw, fs=fs, low=20.0, high=450.0, axis=0).astype(np.float32)

        raw_parts.append(x_raw)
        filtered_parts.append(x_filtered)
        time_parts.append(time_ms)
        class_parts.append(sample_labels)
        file_parts.append(np.full(len(df), file_idx, dtype=np.int16))
        per_file.append(
            {
                "file": path.name,
                "rows": int(len(df)),
                "estimated_fs_hz": fs,
                "class_counts": {str(k): int(v) for k, v in df["class"].value_counts().sort_index().items()},
            }
        )

        win_samples = int(round(win_ms * fs / 1000.0))
        stride_samples = int(round(stride_ms * fs / 1000.0))
        segment_id = 0
        for start, stop, class_id in contiguous_segments(sample_labels):
            if class_id == 0 or stop - start < win_samples:
                segment_id += 1
                continue
            for window_start in range(start, stop - win_samples + 1, stride_samples):
                window_stop = window_start + win_samples
                windows.append(x_filtered[window_start:window_stop].T)
                rows.append(
                    {
                        "dataset": "uci_emg_data_for_gestures",
                        "subject": "01",
                        "file": path.name,
                        "file_index": file_idx,
                        "segment_id": segment_id,
                        "sample_start": window_start,
                        "sample_stop": window_stop,
                        "start_ms": int(time_ms[window_start]),
                        "stop_ms": int(time_ms[window_stop - 1]),
                        "class_id": class_id,
                        "label_name": EMG_CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    }
                )
            segment_id += 1

    x_raw_all = np.concatenate(raw_parts, axis=0)
    x_filtered_all = np.concatenate(filtered_parts, axis=0)
    time_all = np.concatenate(time_parts)
    class_all = np.concatenate(class_parts)
    file_all = np.concatenate(file_parts)
    windows_arr = np.stack(windows).astype(np.float32)
    labels = np.asarray([row["label_name"] for row in rows])
    class_ids = np.asarray([row["class_id"] for row in rows], dtype=np.int16)
    file_indices = np.asarray([row["file_index"] for row in rows], dtype=np.int16)
    features = extract_emg_features(windows_arr)
    names = emg_feature_names(channels)

    np.savez_compressed(
        out / "step1_emg_raw_subject01.npz",
        x_raw=x_raw_all,
        time_ms=time_all,
        sample_class=class_all,
        file_index=file_all,
        channels=np.asarray(channels),
        files=np.asarray(file_names),
    )
    np.savez_compressed(
        out / "step2_emg_bandpass_20_450hz.npz",
        x_filtered=x_filtered_all,
        time_ms=time_all,
        sample_class=class_all,
        file_index=file_all,
        fs_hz=np.asarray(fs_values, dtype=np.float32),
        channels=np.asarray(channels),
    )
    np.savez_compressed(
        out / "step3_emg_windows_200ms.npz",
        windows=windows_arr,
        labels=labels,
        class_ids=class_ids,
        file_index=file_indices,
        fs_hz=np.asarray(float(np.median(fs_values)), dtype=np.float32),
        channels=np.asarray(channels),
        window_ms=np.asarray(win_ms, dtype=np.float32),
        stride_ms=np.asarray(stride_ms, dtype=np.float32),
    )
    np.savez_compressed(
        out / "step4_emg_time_domain_features.npz",
        features=features,
        labels=labels,
        class_ids=class_ids,
        file_index=file_indices,
        feature_names=names,
    )
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)

    loo_preds, loo_summary = loo_nearest_centroid(features, labels)
    decode_rows = [dict(row, predicted_label=pred, correct=bool(pred == row["label_name"])) for row, pred in zip(rows, loo_preds)]
    pd.DataFrame(decode_rows).to_csv(out / "decode_sanity_loo.csv", index=False)
    save_json(out / "decode_sanity_loo.json", loo_summary)

    train_mask = file_indices == 0
    test_mask = file_indices == 1
    session_preds = nearest_centroid_predict(features[train_mask], labels[train_mask], features[test_mask])
    session_summary = classification_summary(labels[test_mask], session_preds)
    test_rows = [dict(row, predicted_label=pred, correct=bool(pred == row["label_name"])) for row, pred in zip(np.asarray(rows, dtype=object)[test_mask], session_preds)]
    pd.DataFrame(test_rows).to_csv(out / "decode_session1_to_session2.csv", index=False)
    save_json(out / "decode_session1_to_session2.json", session_summary)

    save_json(
        out / "summary.json",
        {
            "method": "Surface EMG gesture pipeline: estimate 1 kHz sampling from millisecond timestamps, bandpass 20-450 Hz, split nonzero gesture labels into 200 ms windows, extract time-domain features, nearest-centroid sanity decode.",
            "source": "UCI EMG Data for Gestures, subject 01, two recordings.",
            "raw_shape": list(x_raw_all.shape),
            "bandpass_shape": list(x_filtered_all.shape),
            "windows_shape": list(windows_arr.shape),
            "features_shape": list(features.shape),
            "channels": channels,
            "window_ms": win_ms,
            "stride_ms": stride_ms,
            "class_names": {str(k): v for k, v in EMG_CLASS_NAMES.items()},
            "classes_used_for_windows": sorted(np.unique(labels).tolist()),
            "files": per_file,
            "loo_decode": loo_summary,
            "session1_to_session2_decode": session_summary,
        },
    )


NIFTI_DTYPES = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    64: "f8",
    256: "i1",
    512: "u2",
    768: "u4",
}


def read_nifti(path: Path) -> tuple[np.ndarray, dict]:
    raw = path.read_bytes()
    sizeof_hdr_le = struct.unpack("<i", raw[:4])[0]
    sizeof_hdr_be = struct.unpack(">i", raw[:4])[0]
    if sizeof_hdr_le == 348:
        endian = "<"
    elif sizeof_hdr_be == 348:
        endian = ">"
    else:
        raise ValueError(f"{path} is not a NIfTI-1 single-file image")

    dim = struct.unpack(endian + "8h", raw[40:56])
    ndim = int(dim[0])
    shape = tuple(int(x) for x in dim[1 : ndim + 1])
    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    bitpix = struct.unpack(endian + "h", raw[72:74])[0]
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    vox_offset = int(round(struct.unpack(endian + "f", raw[108:112])[0]))
    scl_slope = struct.unpack(endian + "f", raw[112:116])[0]
    scl_inter = struct.unpack(endian + "f", raw[116:120])[0]
    dtype_code = NIFTI_DTYPES.get(datatype)
    if dtype_code is None:
        raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")
    dtype = np.dtype(dtype_code).newbyteorder(endian)
    count = math.prod(shape)
    arr = np.frombuffer(raw, dtype=dtype, count=count, offset=vox_offset).reshape(shape, order="F")
    data = np.asarray(arr, dtype=np.float32)
    if np.isfinite(scl_slope) and scl_slope not in {0.0, 1.0}:
        data = data * float(scl_slope)
    if np.isfinite(scl_inter) and scl_inter != 0.0:
        data = data + float(scl_inter)
    header = {
        "path": str(path.relative_to(ROOT)),
        "shape": list(shape),
        "datatype": int(datatype),
        "bitpix": int(bitpix),
        "pixdim": [float(v) for v in pixdim],
        "vox_offset": int(vox_offset),
        "scl_slope": float(scl_slope),
        "scl_inter": float(scl_inter),
    }
    return data, header


def volume_labels_from_events(events_path: Path, n_volumes: int, tr: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    events = pd.read_csv(events_path, sep="\t")
    mids = (np.arange(n_volumes, dtype=np.float32) + 0.5) * tr
    labels = np.full(n_volumes, "rest", dtype=object)
    for _, event in events.iterrows():
        onset = float(event["onset"])
        duration = float(event["duration"])
        trial_type = str(event["trial_type"])
        in_event = (mids >= onset) & (mids < onset + duration)
        labels[in_event] = trial_type
    rows = [
        {
            "volume": int(i),
            "volume_mid_s": float(mids[i]),
            "label_name": str(labels[i]),
        }
        for i in range(n_volumes)
    ]
    return mids, labels.astype(str), rows


def make_grid_roi_timeseries(psc: np.ndarray, coords: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    bins = []
    for axis, size in enumerate(image_shape):
        edges = np.linspace(0, size, 4)
        bins.append(np.minimum(np.searchsorted(edges[1:-1], coords[:, axis], side="right"), 2))
    roi_ids = bins[0] * 9 + bins[1] * 3 + bins[2]
    unique_ids = np.asarray(sorted(np.unique(roi_ids)))
    roi_ts = np.zeros((psc.shape[0], len(unique_ids)), dtype=np.float32)
    roi_names = []
    for out_idx, roi_id in enumerate(unique_ids):
        mask = roi_ids == roi_id
        roi_ts[:, out_idx] = psc[:, mask].mean(axis=1)
        x_bin = roi_id // 9
        y_bin = (roi_id % 9) // 3
        z_bin = roi_id % 3
        roi_names.append(f"grid_x{x_bin}_y{y_bin}_z{z_bin}")
    return roi_ts, np.asarray(roi_names)


def gamma_pdf(t: np.ndarray, shape: float, scale: float = 1.0) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    y = np.zeros_like(t, dtype=np.float32)
    positive = t > 0
    y[positive] = (
        (t[positive] ** (shape - 1.0))
        * np.exp(-t[positive] / scale)
        / ((scale**shape) * math.gamma(shape))
    )
    return y


def canonical_hrf(tr: float, duration_s: float = 35.0) -> np.ndarray:
    times = np.arange(0.0, duration_s + tr, tr, dtype=np.float32)
    hrf = gamma_pdf(times, 6.0) - gamma_pdf(times, 16.0) / 6.0
    hrf = hrf / (np.sum(np.abs(hrf)) + 1e-6)
    return hrf.astype(np.float32)


def fit_roi_glm(roi_ts: np.ndarray, labels: np.ndarray, tr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task = (labels == "listening").astype(np.float32)
    regressor = np.convolve(task, canonical_hrf(tr), mode="full")[: len(task)]
    regressor = (regressor - regressor.mean()) / (regressor.std() + 1e-6)
    linear = np.linspace(-1.0, 1.0, len(task), dtype=np.float32)
    design = np.column_stack([np.ones(len(task), dtype=np.float32), linear, regressor])
    betas = np.linalg.lstsq(design, roi_ts, rcond=None)[0]
    residual = roi_ts - design @ betas
    dof = max(len(task) - design.shape[1], 1)
    sigma2 = np.sum(residual**2, axis=0) / dof
    inv_xtx = np.linalg.pinv(design.T @ design)
    task_se = np.sqrt(np.maximum(sigma2 * inv_xtx[2, 2], 1e-12))
    task_t = betas[2] / task_se
    return design.astype(np.float32), betas.astype(np.float32), task_t.astype(np.float32)


def run_fmri_pipeline() -> None:
    out = PROCESSED / "fmri_moae_pipeline"
    out.mkdir(parents=True, exist_ok=True)
    base = RAW / "fmri_moae" / "MoAEpilot"
    bold_path = base / "sub-01" / "func" / "sub-01_task-auditory_bold.nii"
    events_path = base / "sub-01" / "func" / "sub-01_task-auditory_events.tsv"
    json_path = base / "task-auditory_bold.json"

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    tr = float(meta["RepetitionTime"])
    bold, header = read_nifti(bold_path)
    if bold.ndim != 4:
        raise ValueError(f"Expected a 4D BOLD image, got shape {bold.shape}")

    n_volumes = bold.shape[3]
    mean_img = bold.mean(axis=3)
    positive = mean_img > 0
    threshold = float(np.percentile(mean_img[positive], 40))
    brain_mask = positive & (mean_img >= threshold)
    coords = np.argwhere(brain_mask)
    voxel_ts = bold[brain_mask].T
    baseline = voxel_ts.mean(axis=0, keepdims=True)
    valid = baseline.ravel() > 0
    coords = coords[valid]
    voxel_ts = voxel_ts[:, valid]
    baseline = baseline[:, valid]
    psc = ((voxel_ts - baseline) / baseline) * 100.0
    psc = detrend(psc, axis=0, type="linear").astype(np.float32)
    roi_ts, roi_names = make_grid_roi_timeseries(psc, coords, bold.shape[:3])

    mids, volume_labels, label_rows = volume_labels_from_events(events_path, n_volumes, tr)
    write_csv(out / "volume_labels.csv", label_rows)

    block_size = int(round(42.0 / tr))
    block_rows = []
    block_features = []
    block_labels = []
    for block_id, start in enumerate(range(0, n_volumes, block_size)):
        stop = min(start + block_size, n_volumes)
        if stop - start != block_size:
            continue
        labels, counts = np.unique(volume_labels[start:stop], return_counts=True)
        label = str(labels[int(np.argmax(counts))])
        block_features.append(roi_ts[start:stop].mean(axis=0))
        block_labels.append(label)
        block_rows.append(
            {
                "block_id": block_id,
                "volume_start": start,
                "volume_stop": stop,
                "start_s": float(start * tr),
                "stop_s": float(stop * tr),
                "label_name": label,
            }
        )

    block_features_arr = np.stack(block_features).astype(np.float32)
    block_labels_arr = np.asarray(block_labels)
    write_csv(out / "block_labels.csv", block_rows)
    block_preds, block_summary = loo_nearest_centroid(block_features_arr, block_labels_arr)
    decode_rows = [dict(row, predicted_label=pred, correct=bool(pred == row["label_name"])) for row, pred in zip(block_rows, block_preds)]
    pd.DataFrame(decode_rows).to_csv(out / "decode_sanity_blocks.csv", index=False)

    rest_mean = roi_ts[volume_labels == "rest"].mean(axis=0)
    listening_mean = roi_ts[volume_labels == "listening"].mean(axis=0)
    contrast = listening_mean - rest_mean
    design, betas, task_t = fit_roi_glm(roi_ts, volume_labels, tr)
    top_order = np.argsort(task_t)[::-1]
    top_positive_rois = [
        {
            "roi": str(roi_names[idx]),
            "task_beta": float(betas[2, idx]),
            "task_t": float(task_t[idx]),
        }
        for idx in top_order[:5]
    ]

    save_json(
        out / "step1_nifti_metadata.json",
        {
            "nifti": header,
            "bids_metadata": meta,
            "events_file": str(events_path.relative_to(ROOT)),
            "note": "The BIDS NIfTI contains 84 volumes; the source README describes 96 acquisitions and recommends discarding early scans. The BIDS metadata records 12 volumes discarded by user.",
        },
    )
    np.savez_compressed(
        out / "step2_brain_mask_and_mean.npz",
        mean_image=mean_img.astype(np.float32),
        brain_mask=brain_mask,
        threshold=np.asarray(threshold, dtype=np.float32),
        voxel_coordinates=coords.astype(np.int16),
    )
    np.savez_compressed(
        out / "step3_fmri_roi_timeseries_psc.npz",
        roi_time_series=roi_ts,
        labels=volume_labels,
        volume_mid_s=mids,
        roi_names=roi_names,
        tr=np.asarray(tr, dtype=np.float32),
    )
    np.savez_compressed(
        out / "step4_fmri_block_features.npz",
        features=block_features_arr,
        labels=block_labels_arr,
        roi_names=roi_names,
        block_size_volumes=np.asarray(block_size, dtype=np.int16),
        tr=np.asarray(tr, dtype=np.float32),
    )
    np.savez_compressed(
        out / "step5_fmri_listening_minus_rest_contrast.npz",
        contrast=contrast.astype(np.float32),
        listening_mean=listening_mean.astype(np.float32),
        rest_mean=rest_mean.astype(np.float32),
        roi_names=roi_names,
    )
    np.savez_compressed(
        out / "step6_fmri_roi_glm_features.npz",
        design_matrix=design,
        betas=betas,
        task_t=task_t,
        roi_names=roi_names,
        columns=np.asarray(["intercept", "linear_drift", "listening_hrf"]),
    )
    save_json(out / "decode_sanity_blocks.json", block_summary)
    save_json(
        out / "summary.json",
        {
            "method": "Auditory fMRI teaching pipeline: read NIfTI BOLD volumes, create intensity brain mask, convert masked voxel time series to percent signal change, average into 3x3x3 grid ROI time series, aggregate 42 s block features, nearest-centroid sanity decode.",
            "source": "SPM/UCL MoAEpilot BIDS auditory fMRI dataset.",
            "bold_shape": list(bold.shape),
            "tr_s": tr,
            "brain_mask_shape": list(brain_mask.shape),
            "brain_mask_voxels": int(brain_mask.sum()),
            "roi_time_series_shape": list(roi_ts.shape),
            "block_features_shape": list(block_features_arr.shape),
            "labels": {str(k): int(v) for k, v in pd.Series(volume_labels).value_counts().sort_index().items()},
            "block_labels": {str(k): int(v) for k, v in pd.Series(block_labels_arr).value_counts().sort_index().items()},
            "decode": block_summary,
            "glm_design_shape": list(design.shape),
            "glm_betas_shape": list(betas.shape),
            "glm_top_positive_rois": top_positive_rois,
        },
    )


def main() -> None:
    run_emg_pipeline()
    run_fmri_pipeline()


if __name__ == "__main__":
    main()
