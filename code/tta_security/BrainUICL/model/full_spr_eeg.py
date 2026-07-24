"""Core state and filtering primitives for full SPR on EEG sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from model.spr_eeg import self_centered_clean_probabilities


@dataclass
class SPRDelayedRecord:
    data_path: Path
    pseudo_labels: np.ndarray
    task: int
    subject: int
    sequence_index: int
    is_proxy: bool = False

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.pseudo_labels = np.asarray(self.pseudo_labels, dtype=np.int64).reshape(-1)
        self.task = int(self.task)
        self.subject = int(self.subject)
        self.sequence_index = int(self.sequence_index)
        self.is_proxy = bool(self.is_proxy)

    def state_dict(self) -> dict[str, Any]:
        return {
            "data_path": str(self.data_path),
            "pseudo_labels": self.pseudo_labels.tolist(),
            "task": self.task,
            "subject": self.subject,
            "sequence_index": self.sequence_index,
            "is_proxy": self.is_proxy,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SPRDelayedRecord":
        return cls(**state)


@dataclass
class SPRPurifiedRecord(SPRDelayedRecord):
    epoch_mask: np.ndarray | None = None
    clean_probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.epoch_mask is None or self.clean_probabilities is None:
            raise ValueError("Purified records require mask and clean probabilities")
        self.epoch_mask = np.asarray(self.epoch_mask, dtype=bool).reshape(-1)
        self.clean_probabilities = np.asarray(
            self.clean_probabilities, dtype=np.float64
        ).reshape(-1)
        if self.epoch_mask.shape != self.pseudo_labels.shape:
            raise ValueError("Purified epoch mask must match pseudo labels")
        if self.clean_probabilities.shape != self.pseudo_labels.shape:
            raise ValueError("Purified probabilities must match pseudo labels")
        if not np.isfinite(self.clean_probabilities).all():
            raise ValueError("Purified clean probabilities must be finite")

    @property
    def retained_epochs(self) -> int:
        return int(self.epoch_mask.sum())

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "epoch_mask": self.epoch_mask.tolist(),
                "clean_probabilities": self.clean_probabilities.tolist(),
            }
        )
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SPRPurifiedRecord":
        return cls(**state)


class SPRDelayedBuffer:
    def __init__(self, capacity_sequences: int):
        if capacity_sequences < 1:
            raise ValueError("Delayed Buffer capacity must be positive")
        self.capacity_sequences = int(capacity_sequences)
        self.records: list[SPRDelayedRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    @property
    def is_full(self) -> bool:
        return len(self.records) >= self.capacity_sequences

    def add(self, record: SPRDelayedRecord) -> bool:
        if self.is_full:
            raise RuntimeError("Delayed Buffer must be flushed before another insert")
        self.records.append(record)
        return self.is_full

    def drain(self) -> list[SPRDelayedRecord]:
        records = self.records
        self.records = []
        return records

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity_sequences": self.capacity_sequences,
            "records": [record.state_dict() for record in self.records],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SPRDelayedBuffer":
        buffer = cls(state["capacity_sequences"])
        buffer.records = [
            SPRDelayedRecord.from_state_dict(record) for record in state["records"]
        ]
        return buffer


class SPRPurifiedMemory:
    """Class-aware fixed memory accounted in retained 30-second epochs."""

    def __init__(self, capacity_epochs: int, num_classes: int, seed: int):
        if capacity_epochs < 1 or num_classes < 2:
            raise ValueError("Invalid Purified Buffer dimensions")
        self.capacity_epochs = int(capacity_epochs)
        self.num_classes = int(num_classes)
        self.records: list[SPRPurifiedRecord] = []
        self.stream_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.total_replay_draws = 0
        self.proxy_replay_draws = 0

    def __len__(self) -> int:
        return sum(record.retained_epochs for record in self.records)

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for record in self.records:
            labels = record.pseudo_labels[record.epoch_mask]
            counts += np.bincount(labels, minlength=self.num_classes)
        return counts

    def _evict_one(self) -> None:
        counts = self.class_counts().astype(np.float64)
        target = self.capacity_epochs * self.stream_counts / max(
            int(self.stream_counts.sum()), 1
        )
        evict_class = int(np.argmax(counts - target))
        candidates: list[tuple[float, int, int]] = []
        for record_index, record in enumerate(self.records):
            for epoch_index in np.flatnonzero(
                record.epoch_mask & (record.pseudo_labels == evict_class)
            ):
                candidates.append(
                    (
                        float(record.clean_probabilities[epoch_index]),
                        record_index,
                        int(epoch_index),
                    )
                )
        if not candidates:
            raise RuntimeError("Purified Buffer cannot find an epoch to evict")
        _probability, record_index, epoch_index = min(candidates)
        self.records[record_index].epoch_mask[epoch_index] = False
        if self.records[record_index].retained_epochs == 0:
            self.records.pop(record_index)

    def add(
        self,
        delayed: Sequence[SPRDelayedRecord],
        epoch_masks: Sequence[np.ndarray],
        clean_probabilities: Sequence[np.ndarray],
    ) -> dict[str, Any]:
        if not (
            len(delayed) == len(epoch_masks) == len(clean_probabilities)
        ):
            raise ValueError("Purified admission arrays must align")
        before = len(self)
        accepted = 0
        for source, mask_value, probability_value in zip(
            delayed, epoch_masks, clean_probabilities
        ):
            mask = np.asarray(mask_value, dtype=bool).reshape(-1)
            probabilities = np.asarray(probability_value, dtype=np.float64).reshape(-1)
            if mask.shape != source.pseudo_labels.shape or probabilities.shape != mask.shape:
                raise ValueError("Purified admission shape mismatch")
            labels = source.pseudo_labels[mask]
            if labels.size:
                if np.any((labels < 0) | (labels >= self.num_classes)):
                    raise ValueError("Purified admission contains an invalid pseudo class")
                self.stream_counts += np.bincount(labels, minlength=self.num_classes)
                self.records.append(
                    SPRPurifiedRecord(
                        data_path=source.data_path,
                        pseudo_labels=source.pseudo_labels.copy(),
                        task=source.task,
                        subject=source.subject,
                        sequence_index=source.sequence_index,
                        is_proxy=source.is_proxy,
                        epoch_mask=mask.copy(),
                        clean_probabilities=probabilities.copy(),
                    )
                )
                accepted += int(mask.sum())
            while len(self) > self.capacity_epochs:
                self._evict_one()
        return {
            "retained_before": before,
            "accepted_epochs": accepted,
            "retained_after": len(self),
            "evicted_epochs": max(before + accepted - len(self), 0),
            "class_counts": self.class_counts().astype(int).tolist(),
        }

    def sample_records(self, count: int) -> list[SPRPurifiedRecord]:
        if count <= 0 or not self.records:
            return []
        indices = self.rng.choice(
            len(self.records), size=count, replace=len(self.records) < count
        )
        selected = [self.records[int(index)] for index in indices]
        self.total_replay_draws += len(selected)
        self.proxy_replay_draws += sum(int(record.is_proxy) for record in selected)
        return selected

    def stats(self) -> dict[str, Any]:
        proxy_epochs = sum(
            int(record.epoch_mask.sum())
            for record in self.records
            if record.is_proxy
        )
        return {
            "capacity_epochs": self.capacity_epochs,
            "retained_epochs": len(self),
            "sequences": len(self.records),
            "class_counts": self.class_counts().astype(int).tolist(),
            "proxy_epochs": proxy_epochs,
            "proxy_epoch_fraction": proxy_epochs / max(len(self), 1),
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
            "stream_counts": self.stream_counts.tolist(),
            "rng_state": self.rng.bit_generator.state,
            "total_replay_draws": self.total_replay_draws,
            "proxy_replay_draws": self.proxy_replay_draws,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SPRPurifiedMemory":
        memory = cls(
            state["capacity_epochs"], state["num_classes"], seed=0
        )
        memory.records = [
            SPRPurifiedRecord.from_state_dict(record) for record in state["records"]
        ]
        memory.stream_counts = np.asarray(state["stream_counts"], dtype=np.int64)
        memory.rng.bit_generator.state = state["rng_state"]
        memory.total_replay_draws = int(state["total_replay_draws"])
        memory.proxy_replay_draws = int(state["proxy_replay_draws"])
        return memory


class EEGContrastiveEncoder(nn.Module):
    """BrainUICL EEG encoder with a projection head used only by NT-Xent."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_encoder: nn.Module,
        embedding_dim: int = 512,
        projection_hidden: int = 256,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_encoder = feature_encoder
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, projection_dim),
        )

    def epoch_embeddings(self, eog: torch.Tensor, eeg: torch.Tensor, args) -> torch.Tensor:
        batch = eeg.shape[0]
        eog = eog.reshape(
            -1, args.model_param.EogNum, args.model_param.EpochLength
        )
        eeg = eeg.reshape(
            -1, args.model_param.EegNum, args.model_param.EpochLength
        )
        features = self.feature_extractor(eeg, eog)
        features = self.feature_encoder(features)
        return features.reshape(batch, args.model_param.SeqLength, -1)

    def projected_epochs(self, eog: torch.Tensor, eeg: torch.Tensor, args) -> torch.Tensor:
        embeddings = self.epoch_embeddings(eog, eeg, args)
        return self.projector(embeddings.reshape(-1, embeddings.shape[-1]))


def augment_eeg_views(
    eog: torch.Tensor,
    eeg: torch.Tensor,
    *,
    jitter: float,
    scale: float,
    mask_ratio: float,
    channel_drop: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    eog_aug, eeg_aug = eog.clone(), eeg.clone()
    for signal in (eog_aug, eeg_aug):
        batch = signal.shape[0]
        if scale > 0:
            multiplier = torch.empty(
                (batch, 1, 1, 1), device=signal.device
            ).uniform_(1.0 - scale, 1.0 + scale)
            signal.mul_(multiplier)
        if jitter > 0:
            amplitude = signal.std(dim=-1, keepdim=True).clamp_min(1e-6)
            signal.add_(torch.randn_like(signal) * amplitude * jitter)
        if channel_drop > 0:
            keep = torch.rand(
                (batch, 1, signal.shape[2], 1), device=signal.device
            ) >= channel_drop
            signal.mul_(keep)
        width = int(signal.shape[-1] * mask_ratio)
        if width > 0:
            starts = torch.randint(
                0, signal.shape[-1] - width + 1, (batch,), device=signal.device
            )
            for row, start in enumerate(starts.tolist()):
                signal[row, :, :, start : start + width] = 0
    return eog_aug, eeg_aug


def self_centered_admission(
    features: np.ndarray,
    delayed: Sequence[SPRDelayedRecord],
    *,
    ensembles: int,
    bmm_iters: int,
    graph_seed: int,
    admission_rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    if not delayed:
        raise ValueError("Cannot filter an empty Delayed Buffer")
    labels = np.stack([record.pseudo_labels for record in delayed])
    features = np.asarray(features, dtype=np.float32)
    if features.shape[:2] != labels.shape:
        raise ValueError("Expert feature and delayed pseudo-label shapes differ")
    probabilities = self_centered_clean_probabilities(
        features.reshape(-1, features.shape[-1]),
        labels.reshape(-1),
        ensembles=ensembles,
        bmm_iters=bmm_iters,
        seed=graph_seed,
    ).reshape(labels.shape)
    masks = probabilities > admission_rng.random(probabilities.shape)
    return (
        [row.copy() for row in masks],
        [row.copy() for row in probabilities],
        {
            "delayed_sequences": len(delayed),
            "delayed_epochs": int(labels.size),
            "accepted_epochs": int(masks.sum()),
            "acceptance_rate": float(masks.mean()),
            "mean_clean_probability": float(probabilities.mean()),
            "pseudo_class_counts": np.bincount(
                labels.reshape(-1), minlength=int(labels.max()) + 1
            ).astype(int).tolist(),
        },
    )


__all__ = [
    "EEGContrastiveEncoder",
    "SPRDelayedBuffer",
    "SPRDelayedRecord",
    "SPRPurifiedMemory",
    "SPRPurifiedRecord",
    "augment_eeg_views",
    "self_centered_admission",
]
