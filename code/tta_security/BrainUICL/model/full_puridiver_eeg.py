"""Dynamic memory and C/R/U partitioning for full PuriDivER-EEG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class PuriSequenceRecord:
    data_path: Path
    pseudo_labels: np.ndarray
    epoch_mask: np.ndarray
    task: int
    subject: int
    sequence_index: int
    is_proxy: bool = False

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.pseudo_labels = np.asarray(self.pseudo_labels, dtype=np.int64).reshape(-1)
        self.epoch_mask = np.asarray(self.epoch_mask, dtype=bool).reshape(-1)
        if self.epoch_mask.shape != self.pseudo_labels.shape:
            raise ValueError("PuriDivER epoch mask must match pseudo labels")
        self.task = int(self.task)
        self.subject = int(self.subject)
        self.sequence_index = int(self.sequence_index)
        self.is_proxy = bool(self.is_proxy)

    @property
    def retained_epochs(self) -> int:
        return int(self.epoch_mask.sum())

    def state_dict(self) -> dict[str, Any]:
        return {
            "data_path": str(self.data_path),
            "pseudo_labels": self.pseudo_labels.tolist(),
            "epoch_mask": self.epoch_mask.tolist(),
            "task": self.task,
            "subject": self.subject,
            "sequence_index": self.sequence_index,
            "is_proxy": self.is_proxy,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PuriSequenceRecord":
        return cls(**state)


@dataclass(frozen=True)
class PuriMemoryScores:
    losses: np.ndarray
    features: np.ndarray
    classifier_weights: np.ndarray


@dataclass(frozen=True)
class CRUPartition:
    clean_mask: np.ndarray
    relabel_mask: np.ndarray
    unlabeled_mask: np.ndarray
    clean_probability: np.ndarray
    snapshot_probability: np.ndarray
    diagnostics: dict[str, Any]

    def validate(self) -> None:
        masks = np.stack(
            [self.clean_mask, self.relabel_mask, self.unlabeled_mask]
        ).astype(np.int64)
        if not np.all(masks.sum(axis=0) == 1):
            raise ValueError("C/R/U masks must be exhaustive and disjoint")


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def _fit_low_component(
    values: np.ndarray,
    *,
    seed: int,
    min_samples: int,
) -> tuple[np.ndarray | None, np.ndarray | None, list[float] | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        values.size < min_samples
        or not np.isfinite(values).all()
        or float(values.std()) < 1e-8
    ):
        return None, None, None
    gmm = GaussianMixture(
        n_components=2,
        max_iter=50,
        tol=1e-3,
        reg_covar=5e-4,
        random_state=seed,
    ).fit(values[:, None])
    means = gmm.means_.reshape(-1)
    low = int(np.argmin(means))
    posterior = gmm.predict_proba(values[:, None])[:, low]
    assignment = gmm.predict(values[:, None]) == low
    return posterior.astype(np.float32), assignment, sorted(means.astype(float).tolist())


def build_cru_partition(
    losses: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int,
    min_gmm_samples: int = 8,
) -> CRUPartition:
    losses = np.asarray(losses, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != losses.size:
        raise ValueError("PuriDivER losses/probabilities are not aligned")
    if losses.size == 0:
        raise ValueError("PuriDivER cannot partition an empty memory")
    loss_range = float(losses.max() - losses.min())
    normalized = (losses - losses.min()) / max(loss_range, 1e-12)
    clean_probability, clean_mask, loss_means = _fit_low_component(
        normalized, seed=seed, min_samples=min_gmm_samples
    )
    loss_fallback = clean_mask is None
    if clean_mask is None or clean_probability is None:
        clean_mask = np.ones(losses.size, dtype=bool)
        clean_probability = np.ones(losses.size, dtype=np.float32)

    noisy_indices = np.flatnonzero(~clean_mask)
    relabel_mask = np.zeros(losses.size, dtype=bool)
    unlabeled_mask = np.zeros(losses.size, dtype=bool)
    uncertainty_means = None
    uncertainty_fallback = False
    if noisy_indices.size:
        uncertainty = 1.0 - probabilities.max(axis=1)
        _posterior, low_uncertainty, uncertainty_means = _fit_low_component(
            uncertainty[noisy_indices],
            seed=seed + 1,
            min_samples=min_gmm_samples,
        )
        if low_uncertainty is None:
            uncertainty_fallback = True
            unlabeled_mask[noisy_indices] = True
        else:
            relabel_mask[noisy_indices[low_uncertainty]] = True
            unlabeled_mask[noisy_indices[~low_uncertainty]] = True

    partition = CRUPartition(
        clean_mask=np.asarray(clean_mask, dtype=bool),
        relabel_mask=relabel_mask,
        unlabeled_mask=unlabeled_mask,
        clean_probability=np.asarray(clean_probability, dtype=np.float32),
        snapshot_probability=probabilities.astype(np.float32),
        diagnostics={
            "loss_gmm_means": loss_means,
            "uncertainty_gmm_means": uncertainty_means,
            "loss_gmm_fallback_all_clean": loss_fallback,
            "uncertainty_gmm_fallback_all_unlabeled": uncertainty_fallback,
            "clean_count": int(np.asarray(clean_mask).sum()),
            "relabel_count": int(relabel_mask.sum()),
            "unlabeled_count": int(unlabeled_mask.sum()),
            "memory_epochs": int(losses.size),
        },
    )
    partition.validate()
    return partition


class DynamicPuriMemory:
    """Full-sequence storage with epoch-level dynamic purity/diversity pruning."""

    def __init__(self, capacity_epochs: int, num_classes: int, seed: int):
        if capacity_epochs < 1 or num_classes < 2:
            raise ValueError("Invalid PuriDivER memory dimensions")
        self.capacity_epochs = int(capacity_epochs)
        self.num_classes = int(num_classes)
        self.records: list[PuriSequenceRecord] = []
        self.rng = np.random.default_rng(seed)
        self.total_replay_draws = 0
        self.proxy_replay_draws = 0
        self.total_candidates_seen = 0

    def __len__(self) -> int:
        return sum(record.retained_epochs for record in self.records)

    def active_refs(self) -> list[tuple[int, int]]:
        return [
            (record_index, int(epoch_index))
            for record_index, record in enumerate(self.records)
            for epoch_index in np.flatnonzero(record.epoch_mask)
        ]

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for record in self.records:
            counts += np.bincount(
                record.pseudo_labels[record.epoch_mask], minlength=self.num_classes
            )
        return counts

    def update(
        self,
        incoming: Sequence[PuriSequenceRecord],
        scorer: Callable[[Sequence[PuriSequenceRecord]], PuriMemoryScores],
        diversity_coefficient: float,
    ) -> dict[str, Any]:
        if not 0.0 <= diversity_coefficient <= 1.0:
            raise ValueError("PuriDivER diversity coefficient must be in [0, 1]")
        incoming_epochs = sum(record.retained_epochs for record in incoming)
        self.total_candidates_seen += incoming_epochs
        self.records.extend(incoming)
        candidate_epochs = len(self)
        if candidate_epochs <= self.capacity_epochs:
            return {
                "incoming_epochs": incoming_epochs,
                "candidate_epochs": candidate_epochs,
                "removed_epochs": 0,
                "retained_epochs": len(self),
                "score_recomputations": 0,
                "class_counts": self.class_counts().astype(int).tolist(),
            }

        refs = self.active_refs()
        scores = scorer(self.records)
        losses = np.asarray(scores.losses, dtype=np.float64).reshape(-1)
        features = np.asarray(scores.features, dtype=np.float64)
        classifier_weights = np.asarray(scores.classifier_weights, dtype=np.float64)
        if losses.shape[0] != len(refs) or features.shape[0] != len(refs):
            raise ValueError("PuriDivER scorer output does not match memory epochs")
        if classifier_weights.shape[1] != features.shape[1]:
            raise ValueError("PuriDivER classifier/feature dimensions differ")

        labels = np.asarray(
            [self.records[r].pseudo_labels[e] for r, e in refs], dtype=np.int64
        )
        active = np.ones(len(refs), dtype=bool)
        removed_scores: list[float] = []
        recomputations = 0
        while int(active.sum()) > self.capacity_epochs:
            counts = np.bincount(labels[active], minlength=self.num_classes)
            majority = np.flatnonzero(counts == counts.max())
            class_id = int(self.rng.choice(majority))
            class_indices = np.flatnonzero(active & (labels == class_id))
            mean_weight = classifier_weights.mean(axis=0)
            relevant = classifier_weights[class_id] > mean_weight
            if not relevant.any():
                relevant = np.ones(features.shape[1], dtype=bool)
            class_features = features[class_indices][:, relevant]
            norms = np.linalg.norm(class_features, axis=1, keepdims=True)
            normalized_features = class_features / np.maximum(norms, 1e-12)
            similarity = normalized_features @ normalized_features.mean(axis=0)
            combined = (
                (1.0 - diversity_coefficient) * _zscore(losses[class_indices])
                + diversity_coefficient * _zscore(similarity)
            )
            local_drop = int(np.argmax(combined))
            drop_index = int(class_indices[local_drop])
            active[drop_index] = False
            removed_scores.append(float(combined[local_drop]))
            recomputations += 1

        for keep, (record_index, epoch_index) in zip(active, refs):
            if not keep:
                self.records[record_index].epoch_mask[epoch_index] = False
        self.records = [record for record in self.records if record.retained_epochs]
        return {
            "incoming_epochs": incoming_epochs,
            "candidate_epochs": candidate_epochs,
            "removed_epochs": len(removed_scores),
            "retained_epochs": len(self),
            "score_recomputations": recomputations,
            "class_counts": self.class_counts().astype(int).tolist(),
            "mean_removed_score": float(np.mean(removed_scores)),
        }

    def sample_epoch_weights(
        self, count: int
    ) -> list[tuple[PuriSequenceRecord, np.ndarray]]:
        refs = self.active_refs()
        if count <= 0 or not refs:
            return []
        selected = self.rng.choice(
            len(refs), size=count, replace=len(refs) < count
        )
        grouped: dict[int, np.ndarray] = {}
        proxy_draws = 0
        for selected_index in selected:
            record_index, epoch_index = refs[int(selected_index)]
            grouped.setdefault(
                record_index,
                np.zeros_like(self.records[record_index].pseudo_labels, dtype=np.int64),
            )[epoch_index] += 1
            proxy_draws += int(self.records[record_index].is_proxy)
        self.total_replay_draws += int(count)
        self.proxy_replay_draws += proxy_draws
        return [(self.records[index], weights) for index, weights in grouped.items()]

    def stats(self) -> dict[str, Any]:
        proxy_epochs = sum(
            record.retained_epochs for record in self.records if record.is_proxy
        )
        return {
            "capacity_epochs": self.capacity_epochs,
            "retained_epochs": len(self),
            "sequences": len(self.records),
            "class_counts": self.class_counts().astype(int).tolist(),
            "proxy_epochs": proxy_epochs,
            "proxy_epoch_fraction": proxy_epochs / max(len(self), 1),
            "total_candidates_seen": self.total_candidates_seen,
            "total_replay_draws": self.total_replay_draws,
            "proxy_replay_draws": self.proxy_replay_draws,
            "proxy_replay_fraction": self.proxy_replay_draws
            / max(self.total_replay_draws, 1),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity_epochs": self.capacity_epochs,
            "num_classes": self.num_classes,
            "records": [record.state_dict() for record in self.records],
            "rng_state": self.rng.bit_generator.state,
            "total_replay_draws": self.total_replay_draws,
            "proxy_replay_draws": self.proxy_replay_draws,
            "total_candidates_seen": self.total_candidates_seen,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DynamicPuriMemory":
        memory = cls(state["capacity_epochs"], state["num_classes"], seed=0)
        memory.records = [
            PuriSequenceRecord.from_state_dict(record) for record in state["records"]
        ]
        memory.rng.bit_generator.state = state["rng_state"]
        memory.total_replay_draws = int(state["total_replay_draws"])
        memory.proxy_replay_draws = int(state["proxy_replay_draws"])
        memory.total_candidates_seen = int(state["total_candidates_seen"])
        return memory


__all__ = [
    "CRUPartition",
    "DynamicPuriMemory",
    "PuriMemoryScores",
    "PuriSequenceRecord",
    "build_cru_partition",
]
