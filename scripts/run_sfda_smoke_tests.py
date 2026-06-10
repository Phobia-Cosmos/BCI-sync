#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "sfda_smoke"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def classification_summary(labels: np.ndarray, preds: np.ndarray) -> dict:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    per_class = {}
    for cls in np.unique(labels):
        mask = labels == cls
        per_class[str(cls)] = float(np.mean(preds[mask] == labels[mask]))
    return {
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "per_class_accuracy": per_class,
        "n": int(len(labels)),
    }


def train_centroids(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(sorted(np.unique(labels)))
    centroids = np.stack([features[labels == cls].mean(axis=0) for cls in classes])
    return classes, centroids


def centroid_logits(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    d2 = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    scale = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    return -d2 / max(scale, 1e-6)


def predict_centroids(
    features: np.ndarray,
    classes: np.ndarray,
    centroids: np.ndarray,
    bias: np.ndarray | None = None,
    logit_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    logits = centroid_logits(features, centroids) * logit_scale
    if bias is not None:
        logits = logits + bias[None, :]
    probs = softmax(logits)
    preds = classes[np.argmax(probs, axis=1)]
    return preds, probs


def spd_logm(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.maximum(vals, 1e-7)
    return (vecs * np.log(vals)[None, :]) @ vecs.T


def covariance_log_features(epochs: np.ndarray) -> np.ndarray:
    features = []
    iu = np.triu_indices(epochs.shape[1])
    eye = np.eye(epochs.shape[1], dtype=np.float32)
    for epoch in epochs:
        centered = epoch - epoch.mean(axis=1, keepdims=True)
        cov = centered @ centered.T / max(centered.shape[1] - 1, 1)
        cov = cov + eye * (1e-4 * np.trace(cov) / cov.shape[0] + 1e-6)
        features.append(spd_logm(cov)[iu])
    return np.asarray(features, dtype=np.float32)


def im_bias_grid_search(logits: np.ndarray) -> tuple[np.ndarray, dict]:
    # Binary grid search is enough for the MI example and keeps this script dependency-free.
    if logits.shape[1] != 2:
        return np.zeros(logits.shape[1], dtype=np.float32), {"note": "only binary grid search implemented"}
    best = None
    for delta in np.linspace(-4.0, 4.0, 401):
        bias = np.asarray([-delta / 2.0, delta / 2.0], dtype=np.float32)
        probs = softmax(logits + bias[None, :])
        cond_entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1))
        marginal = probs.mean(axis=0)
        marginal_entropy = -np.sum(marginal * np.log(marginal + 1e-8))
        loss = cond_entropy - marginal_entropy
        row = (float(loss), float(delta), bias, float(cond_entropy), float(marginal_entropy), marginal)
        if best is None or row[0] < best[0]:
            best = row
    assert best is not None
    return best[2], {
        "objective": "conditional_entropy - marginal_entropy",
        "delta": best[1],
        "conditional_entropy": best[3],
        "marginal_entropy": best[4],
        "mean_prediction_probability": [float(x) for x in best[5]],
    }


def run_spdim_like_mi() -> dict:
    out = OUT / "spdim_like_mi"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(PROCESSED / "mi_eeg_s001_pipeline" / "step_mi_epochs_8_30hz.npz", allow_pickle=True)
    epochs = z["x"].astype(np.float32)
    labels = z["labels"].astype(str)
    meta = pd.read_csv(PROCESSED / "mi_eeg_s001_pipeline" / "labels.csv")
    features = covariance_log_features(epochs)

    source_mask = meta["task"].to_numpy() == "executed_left_right_fist"
    target_mask = meta["task"].to_numpy() == "imagined_left_right_fist"
    x_source = features[source_mask]
    y_source = labels[source_mask]
    x_target_full = features[target_mask]
    y_target_full = labels[target_mask]

    def run_target_case(case_name: str, keep_mask: np.ndarray) -> dict:
        x_target = x_target_full[keep_mask]
        y_target = y_target_full[keep_mask]
        classes, source_centroids = train_centroids(x_source, y_source)
        source_preds, source_probs = predict_centroids(x_target, classes, source_centroids)

        # Log-Euclidean recentering approximates the Riemannian mean alignment step.
        source_mean = x_source.mean(axis=0, keepdims=True)
        target_mean = x_target.mean(axis=0, keepdims=True)
        x_target_recentered = x_target - target_mean + source_mean
        rct_logits = centroid_logits(x_target_recentered, source_centroids)
        rct_probs = softmax(rct_logits)
        rct_preds = classes[np.argmax(rct_probs, axis=1)]

        bias, bias_info = im_bias_grid_search(rct_logits)
        im_probs = softmax(rct_logits + bias[None, :])
        im_preds = classes[np.argmax(im_probs, axis=1)]

        rows = []
        for idx, true_label in enumerate(y_target):
            rows.append(
                {
                    "case": case_name,
                    "sample": idx,
                    "true_label_hidden_during_adaptation": true_label,
                    "source_only_pred": source_preds[idx],
                    "rct_pred": rct_preds[idx],
                    "rct_im_bias_pred": im_preds[idx],
                    "source_only_confidence": float(source_probs[idx].max()),
                    "rct_confidence": float(rct_probs[idx].max()),
                    "rct_im_bias_confidence": float(im_probs[idx].max()),
                }
            )
        pd.DataFrame(rows).to_csv(out / f"predictions_{case_name}.csv", index=False)
        return {
            "target_case": case_name,
            "target_label_counts": {str(k): int(v) for k, v in pd.Series(y_target).value_counts().sort_index().items()},
            "source_only": classification_summary(y_target, source_preds),
            "rct_log_euclidean": classification_summary(y_target, rct_preds),
            "rct_plus_information_maximization_bias": classification_summary(y_target, im_preds),
            "im_bias": [float(v) for v in bias],
            "im_bias_info": bias_info,
        }

    full_keep = np.ones(len(y_target_full), dtype=bool)
    shifted_keep = np.zeros(len(y_target_full), dtype=bool)
    # Artificial target label shift: keep all left_fist examples and only three right_fist examples.
    right_seen = 0
    for idx, label in enumerate(y_target_full):
        if label == "left_fist":
            shifted_keep[idx] = True
        elif label == "right_fist" and right_seen < 3:
            shifted_keep[idx] = True
            right_seen += 1

    np.savez_compressed(
        out / "step1_mi_covariance_log_features.npz",
        features=features,
        labels=labels,
        tasks=meta["task"].to_numpy(dtype=str),
        feature_type=np.asarray("upper-triangular log covariance"),
    )
    summary = {
        "method_family": "SPDIM/TSM-inspired smoke test, not the official SPDIM implementation.",
        "source_domain": "PhysioNet EEGMMI S001 executed_left_right_fist",
        "target_domain": "PhysioNet EEGMMI S001 imagined_left_right_fist",
        "source_feature_shape": list(x_source.shape),
        "target_full_feature_shape": list(x_target_full.shape),
        "source_label_counts": {str(k): int(v) for k, v in pd.Series(y_source).value_counts().sort_index().items()},
        "cases": [
            run_target_case("full_target", full_keep),
            run_target_case("artificial_label_shift", shifted_keep),
        ],
        "adaptation_inputs": {
            "source_training": "source labels are used only to fit centroids/source model.",
            "target_adaptation": "target labels are hidden; only target covariance features are used for recentering and IM bias search.",
            "evaluation": "target labels are used after adaptation only for sanity metrics.",
        },
    }
    save_json(out / "summary.json", summary)
    return summary


def generate_sleep_sequence_shape_demo() -> dict:
    out = OUT / "sf_uida_sleep_shape_demo" / "ISRUC" / "1"
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "label").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    seq_len, channels, samples = 20, 8, 3000
    labels = np.asarray([0, 0, 1, 2, 2, 2, 3, 3, 2, 4, 4, 2, 2, 1, 0, 0, 4, 2, 3, 0], dtype=np.int64)
    t = np.linspace(0, 30.0, samples, endpoint=False)
    freqs = [9.0, 6.0, 12.0, 2.0, 5.0]
    x = np.zeros((seq_len, channels, samples), dtype=np.float32)
    channel_weights = rng.normal(1.0, 0.15, size=(channels,))
    for i, label in enumerate(labels):
        base = np.sin(2 * np.pi * freqs[int(label)] * t)
        harmonic = 0.4 * np.sin(2 * np.pi * (freqs[int(label)] / 2.0) * t + 0.5)
        noise = rng.normal(0.0, 0.25, size=(channels, samples))
        x[i] = (channel_weights[:, None] * (base + harmonic)[None, :] + noise).astype(np.float32)
    np.save(out / "data" / "0.npy", x)
    np.save(out / "label" / "0.npy", labels)

    eog = x[:, :2, :]
    eeg = x[:, 2:, :]
    summary = {
        "path": str(out.relative_to(ROOT)),
        "official_sf_uida_expected_file_layout": "{subject}/data/{sequence_id}.npy and {subject}/label/{sequence_id}.npy",
        "x_data_shape": list(x.shape),
        "label_shape": list(labels.shape),
        "isruc_split_used_by_official_dataloader": {
            "eog": "x_data[:, :2, :]",
            "eeg": "x_data[:, 2:, :]",
            "eog_shape": list(eog.shape),
            "eeg_shape": list(eeg.shape),
        },
        "class_names": ["W", "N1", "N2", "N3", "REM"],
    }
    save_json(OUT / "sf_uida_sleep_shape_demo" / "summary.json", summary)
    return summary


def run_sf_uida_like_toy() -> dict:
    out = OUT / "sf_uida_like_toy"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    n_classes, dim = 5, 12
    source_counts = np.asarray([30, 12, 45, 25, 28])
    target_counts = np.asarray([15, 5, 35, 20, 25])
    base_centers = rng.normal(0.0, 1.0, size=(n_classes, dim))
    source_x = []
    source_y = []
    target_x = []
    target_y = []
    subject_shift = rng.normal(0.7, 0.15, size=(dim,))
    class_conditional_shift = rng.normal(0.0, 0.25, size=(n_classes, dim))
    for cls in range(n_classes):
        source_x.append(base_centers[cls] + rng.normal(0, 0.55, size=(source_counts[cls], dim)))
        source_y.extend([cls] * int(source_counts[cls]))
        target_x.append(base_centers[cls] + subject_shift + class_conditional_shift[cls] + rng.normal(0, 0.6, size=(target_counts[cls], dim)))
        target_y.extend([cls] * int(target_counts[cls]))
    source_x = np.vstack(source_x).astype(np.float32)
    target_x = np.vstack(target_x).astype(np.float32)
    source_y = np.asarray(source_y)
    target_y = np.asarray(target_y)

    classes, source_centroids = train_centroids(source_x, source_y)
    confidence_scale = 4.0
    source_only_preds, source_only_probs = predict_centroids(target_x, classes, source_centroids, logit_scale=confidence_scale)

    # Stage 1 in spirit: align target marginal PT(x) to the stored source feature statistics.
    source_mean = source_x.mean(axis=0, keepdims=True)
    target_mean = target_x.mean(axis=0, keepdims=True)
    adapted_x = target_x - target_mean + source_mean
    stage1_preds, stage1_probs = predict_centroids(adapted_x, classes, source_centroids, logit_scale=confidence_scale)

    # Stage 2 in spirit: teacher confidence pseudo-labels, then update class prototypes.
    confident = stage1_probs.max(axis=1) >= 0.70
    pseudo = stage1_preds
    personalized_centroids = source_centroids.copy()
    for cls in classes:
        mask = confident & (pseudo == cls)
        if np.any(mask):
            personalized_centroids[int(cls)] = 0.65 * personalized_centroids[int(cls)] + 0.35 * adapted_x[mask].mean(axis=0)
    stage2_preds, stage2_probs = predict_centroids(adapted_x, classes, personalized_centroids, logit_scale=confidence_scale)

    rows = []
    for i, label in enumerate(target_y):
        rows.append(
            {
                "sample": i,
                "true_label_hidden_during_adaptation": int(label),
                "source_only_pred": int(source_only_preds[i]),
                "stage1_marginal_alignment_pred": int(stage1_preds[i]),
                "stage2_pseudo_personalization_pred": int(stage2_preds[i]),
                "stage1_confidence": float(stage1_probs[i].max()),
                "used_for_pseudo_label_update": bool(confident[i]),
            }
        )
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    np.savez_compressed(
        out / "toy_features.npz",
        source_features=source_x,
        source_labels=source_y,
        target_features=target_x,
        target_labels_for_evaluation_only=target_y,
        adapted_target_features=adapted_x,
    )
    summary = {
        "method_family": "SF-UIDA-inspired numpy toy run, not the official PyTorch sleep model.",
        "classes": ["W", "N1", "N2", "N3", "REM"],
        "target_label_counts": {str(k): int(v) for k, v in pd.Series(target_y).value_counts().sort_index().items()},
        "source_only": classification_summary(target_y, source_only_preds),
        "stage1_subject_specific_marginal_alignment": classification_summary(target_y, stage1_preds),
        "stage2_confident_pseudo_label_personalization": classification_summary(target_y, stage2_preds),
        "pseudo_label_threshold": 0.70,
        "toy_logit_confidence_scale": confidence_scale,
        "pseudo_labeled_samples": int(confident.sum()),
        "adaptation_inputs": {
            "source_model_state": "stored source centroids and source feature mean; no source samples used during target adaptation.",
            "target_adaptation": "unlabeled target features only.",
            "evaluation": "target labels used after adaptation only for sanity metrics.",
        },
    }
    save_json(out / "summary.json", summary)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spdim_summary = run_spdim_like_mi()
    shape_summary = generate_sleep_sequence_shape_demo()
    sf_uida_summary = run_sf_uida_like_toy()
    save_json(
        OUT / "summary.json",
        {
            "official_code_status": {
                "sf_uida": {
                    "repo": "code/SF-UIDA",
                    "remote": "https://github.com/xiaobaben/SF-UIDA",
                    "commit": "4172de4",
                    "requires": ["torch", "sklearn", "preprocessed SleepEDF/ISRUC/HMC .npy files", "pretrained checkpoint .pkl files when pretrain=False"],
                    "current_environment": "torch and sklearn are not installed; official full run was not launched.",
                },
                "tsmnet_spdim_family": {
                    "repo": "code/TSMNet",
                    "remote": "https://github.com/rkobler/TSMNet",
                    "commit": "90293b9",
                    "requires": ["torch", "geoopt", "skorch", "hydra", "omegaconf", "mne", "moabb"],
                    "current_environment": "these deep-learning/MOABB dependencies are not installed; official full run was not launched.",
                },
            },
            "smoke_runs": {
                "spdim_like_mi": spdim_summary,
                "sf_uida_sleep_shape_demo": shape_summary,
                "sf_uida_like_toy": sf_uida_summary,
            },
        },
    )


if __name__ == "__main__":
    main()
