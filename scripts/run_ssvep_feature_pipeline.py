#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tsinghua_ssvep"
OUT = ROOT / "data" / "processed" / "ssvep_s1_pipeline"

FS = 250.0
SELECTED_CHANNELS = ["Pz", "PO3", "PO5", "PO4", "PO6", "POz", "O1", "Oz", "O2"]
FILTER_BANKS = [(6, 90), (14, 90), (22, 90), (30, 90), (38, 90)]
N_HARMONICS = 3


def save_json(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_channel_names() -> list[str]:
    names = []
    for line in (RAW / "64-channels.loc").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            names.append(parts[3])
    return names


def make_trials(data: np.ndarray, freqs: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    epochs = []
    rows = []
    for target_idx in range(data.shape[2]):
        for block_idx in range(data.shape[3]):
            epochs.append(data[:, :, target_idx, block_idx])
            rows.append(
                {
                    "trial_index": len(rows),
                    "target_index": target_idx + 1,
                    "block": block_idx + 1,
                    "frequency_hz": float(freqs[target_idx]),
                    "phase_rad": float(phases[target_idx]),
                }
            )
    return np.stack(epochs, axis=0), pd.DataFrame(rows)


def bandpass_epochs(x: np.ndarray, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def cca_first_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    qx, _ = np.linalg.qr(x)
    qy, _ = np.linalg.qr(y)
    singular_values = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(singular_values[0], 0.0, 1.0))


def reference_template(freq: float, phase: float, n_times: int) -> np.ndarray:
    t = np.arange(n_times) / FS
    cols = []
    for harmonic in range(1, N_HARMONICS + 1):
        angle = 2 * np.pi * harmonic * freq * t + harmonic * phase
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.stack(cols, axis=1)


def extract_fbcca_features(filtered_banks: list[np.ndarray], freqs: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_trials = filtered_banks[0].shape[0]
    n_targets = len(freqs)
    n_bands = len(filtered_banks)
    n_times = filtered_banks[0].shape[-1]
    templates = [reference_template(freqs[i], phases[i], n_times) for i in range(n_targets)]
    band_corrs = np.zeros((n_trials, n_bands, n_targets), dtype=np.float32)
    for trial_idx in range(n_trials):
        for band_idx, bank in enumerate(filtered_banks):
            trial = bank[trial_idx].T
            for target_idx, template in enumerate(templates):
                band_corrs[trial_idx, band_idx, target_idx] = cca_first_corr(trial, template)
    band_ids = np.arange(1, n_bands + 1, dtype=np.float32)
    weights = band_ids ** -1.25 + 0.25
    fbcca_scores = np.tensordot(band_corrs**2, weights, axes=([1], [0]))
    return band_corrs, fbcca_scores.astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    mat = sio.loadmat(RAW / "S1.mat")
    data = mat["data"]
    freq_phase = sio.loadmat(RAW / "Freq_Phase.mat")
    freqs = freq_phase["freqs"].ravel()
    phases = freq_phase["phases"].ravel()
    channel_names = load_channel_names()

    save_json(
        "step1_raw_summary.json",
        {
            "source_file": str(RAW / "S1.mat"),
            "variable": "data",
            "shape": list(data.shape),
            "axis_order": ["channel", "time", "target", "block"],
            "sampling_frequency_hz": FS,
            "n_targets": int(data.shape[2]),
            "n_blocks": int(data.shape[3]),
        },
    )

    trials, labels = make_trials(data, freqs, phases)
    channel_lookup = {name.lower(): idx for idx, name in enumerate(channel_names)}
    selected_idx = [channel_lookup[ch.lower()] for ch in SELECTED_CHANNELS]
    selected = trials[:, selected_idx, :]
    labels.to_csv(OUT / "step2_trial_labels.csv", index=False)
    np.savez_compressed(
        OUT / "step2_occipital_epochs.npz",
        x=selected.astype(np.float32),
        target_index=labels["target_index"].to_numpy(np.int16),
        frequency_hz=labels["frequency_hz"].to_numpy(np.float32),
        phase_rad=labels["phase_rad"].to_numpy(np.float32),
        channel_names=np.array(SELECTED_CHANNELS),
        fs=np.array(FS, dtype=np.float32),
    )
    save_json(
        "step2_occipital_epochs_summary.json",
        {
            "shape": list(selected.shape),
            "axis_order": ["trial", "selected_channel", "time"],
            "selected_channels": SELECTED_CHANNELS,
            "n_trials": int(selected.shape[0]),
        },
    )

    stim_start = int(0.5 * FS)
    stim_stop = stim_start + int(5.0 * FS)
    stimulation = selected[:, :, stim_start:stim_stop]
    np.savez_compressed(
        OUT / "step3_stimulation_window.npz",
        x=stimulation.astype(np.float32),
        target_index=labels["target_index"].to_numpy(np.int16),
        channel_names=np.array(SELECTED_CHANNELS),
        fs=np.array(FS, dtype=np.float32),
        window_start_s=np.array(0.5, dtype=np.float32),
        window_duration_s=np.array(5.0, dtype=np.float32),
    )
    save_json(
        "step3_stimulation_window_summary.json",
        {
            "shape": list(stimulation.shape),
            "axis_order": ["trial", "selected_channel", "time"],
            "window_start_s": 0.5,
            "window_duration_s": 5.0,
            "description": "Removed 0.5 s pre-stimulus baseline and kept 5 s flicker stimulation.",
        },
    )

    filtered_banks = []
    for low, high in FILTER_BANKS:
        filtered_banks.append(bandpass_epochs(stimulation, low, high).astype(np.float32))
    filtered = np.stack(filtered_banks, axis=1)
    np.savez_compressed(
        OUT / "step4_filterbank_epochs.npz",
        x=filtered,
        target_index=labels["target_index"].to_numpy(np.int16),
        filter_banks=np.array(FILTER_BANKS, dtype=np.float32),
        channel_names=np.array(SELECTED_CHANNELS),
        fs=np.array(FS, dtype=np.float32),
    )
    save_json(
        "step4_filterbank_epochs_summary.json",
        {
            "shape": list(filtered.shape),
            "axis_order": ["trial", "filter_bank", "selected_channel", "time"],
            "filter_banks_hz": FILTER_BANKS,
        },
    )

    band_corrs, fbcca_scores = extract_fbcca_features(filtered_banks, freqs, phases)
    y_true = labels["target_index"].to_numpy(np.int16)
    y_pred = np.argmax(fbcca_scores, axis=1).astype(np.int16) + 1
    decode = labels.copy()
    decode["predicted_target_index"] = y_pred
    decode["correct"] = y_pred == y_true
    decode.to_csv(OUT / "step6_argmax_decode_sanity.csv", index=False)

    np.savez_compressed(
        OUT / "step5_fbcca_features.npz",
        band_correlations=band_corrs,
        fbcca_scores=fbcca_scores,
        target_index=y_true,
        target_frequencies_hz=freqs.astype(np.float32),
        target_phases_rad=phases.astype(np.float32),
        filter_banks=np.array(FILTER_BANKS, dtype=np.float32),
    )
    feature_preview = pd.DataFrame(
        fbcca_scores[:10, :10],
        columns=[f"score_target_{i + 1}" for i in range(10)],
    )
    feature_preview.insert(0, "true_target_index", y_true[:10])
    feature_preview.to_csv(OUT / "step5_fbcca_feature_preview.csv", index=False)
    save_json(
        "step5_fbcca_features_summary.json",
        {
            "band_correlations_shape": list(band_corrs.shape),
            "fbcca_scores_shape": list(fbcca_scores.shape),
            "axis_order_band_correlations": ["trial", "filter_bank", "candidate_target"],
            "axis_order_fbcca_scores": ["trial", "candidate_target"],
            "feature_definition": "CCA correlation to sine/cosine templates for each target frequency; filter-bank weighted squared correlations are final features.",
            "not_training": True,
        },
    )
    save_json(
        "step6_argmax_decode_sanity_summary.json",
        {
            "description": "Sanity check only: choose candidate target with max FBCCA score.",
            "accuracy": float(np.mean(decode["correct"])),
            "n_trials": int(len(decode)),
            "n_correct": int(decode["correct"].sum()),
        },
    )

    print(f"Saved SSVEP pipeline outputs to {OUT}")
    print(f"step1 raw shape: {data.shape}")
    print(f"step2 selected epochs: {selected.shape}")
    print(f"step3 stimulation window: {stimulation.shape}")
    print(f"step4 filter-bank epochs: {filtered.shape}")
    print(f"step5 band correlations: {band_corrs.shape}")
    print(f"step5 FBCCA scores: {fbcca_scores.shape}")
    print(f"step6 argmax decode sanity accuracy: {np.mean(decode['correct']):.4f}")


if __name__ == "__main__":
    main()
