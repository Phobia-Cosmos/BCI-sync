from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model.spr_eeg import EpochMemoryRecord, self_centered_clean_probabilities


@dataclass(frozen=True)
class PseudoLabelFilterResult:
    records: list[EpochMemoryRecord]
    accepted: np.ndarray
    clean_probabilities: np.ndarray
    metrics: dict


def filter_pseudo_labeled_epochs(
    features: np.ndarray,
    pseudo_labels: np.ndarray,
    data_paths: list[Path],
    *,
    ensembles: int,
    bmm_iters: int,
    seed: int,
    true_labels_for_diagnostics: np.ndarray | None = None,
) -> PseudoLabelFilterResult:
    """Run SCF on every pseudo-labeled epoch without a confidence gate."""

    features = np.asarray(features, dtype=np.float32)
    pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)
    if features.ndim != 3 or pseudo_labels.shape != features.shape[:2]:
        raise ValueError("expected features [N,S,D] and pseudo_labels [N,S]")
    if len(data_paths) != features.shape[0]:
        raise ValueError("data path count does not match sequence count")
    clean_p = self_centered_clean_probabilities(
        features.reshape(-1, features.shape[-1]),
        pseudo_labels.reshape(-1),
        ensembles=ensembles,
        bmm_iters=bmm_iters,
        seed=seed,
    ).reshape(pseudo_labels.shape)
    rng = np.random.default_rng(seed + 1)
    accepted = clean_p > rng.random(clean_p.shape)

    true_labels = None
    if true_labels_for_diagnostics is not None:
        true_labels = np.asarray(true_labels_for_diagnostics, dtype=np.int64)
        if true_labels.shape != pseudo_labels.shape:
            raise ValueError("diagnostic labels do not match pseudo-label shape")

    records = []
    for sequence_index, data_path in enumerate(data_paths):
        for epoch_index in np.flatnonzero(accepted[sequence_index]):
            records.append(
                EpochMemoryRecord(
                    data_path=str(data_path),
                    epoch_index=int(epoch_index),
                    observed_label=int(pseudo_labels[sequence_index, epoch_index]),
                    clean_probability=float(clean_p[sequence_index, epoch_index]),
                    true_label=(
                        int(true_labels[sequence_index, epoch_index])
                        if true_labels is not None
                        else -1
                    ),
                )
            )

    metrics = {
        "candidate_epochs": int(pseudo_labels.size),
        "accepted_epochs": int(accepted.sum()),
        "acceptance_rate": float(accepted.mean()),
        "mean_clean_probability": float(clean_p.mean()),
    }
    if true_labels is not None:
        accepted_count = int(accepted.sum())
        accepted_correct = int(((pseudo_labels == true_labels) & accepted).sum())
        metrics.update(
            {
                "pseudo_label_error_before": float((pseudo_labels != true_labels).mean()),
                "pseudo_label_error_after": (
                    float(1.0 - accepted_correct / accepted_count)
                    if accepted_count
                    else float("nan")
                ),
                "accepted_purity": (
                    float(accepted_correct / accepted_count)
                    if accepted_count
                    else float("nan")
                ),
            }
        )
    return PseudoLabelFilterResult(records, accepted, clean_p, metrics)

