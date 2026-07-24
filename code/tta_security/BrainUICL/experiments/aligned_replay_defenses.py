"""Defense primitives shared by the aligned continual-learning runners.

The helpers in this module deliberately operate on uploaded signal files and
stored pseudo labels only.  Target annotation files are never opened here.
Replay remains sequence based, while SPR and PuriDivER decisions are made at
the epoch level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset

import model.spr_eeg as spr_eeg


def _as_numpy(value: np.ndarray | torch.Tensor, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


@dataclass
class ReplayRecord:
    """One uploaded sequence occurrence and its admission-time metadata."""

    data_path: Path
    pseudo_labels: np.ndarray
    task: int
    subject: int
    sequence_index: int
    poisoned: bool
    repeated_upload: bool
    replay_count: int = 0
    upload_index: int = 0
    uid: str | None = None
    epoch_mask: np.ndarray | None = None
    clean_probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.pseudo_labels = np.asarray(self.pseudo_labels, dtype=np.int64).reshape(-1)
        self.task = int(self.task)
        self.subject = int(self.subject)
        self.sequence_index = int(self.sequence_index)
        self.replay_count = int(self.replay_count)
        self.upload_index = int(self.upload_index)
        if self.uid is None:
            self.uid = (
                f"task-{self.task}:subject-{self.subject}:"
                f"upload-{self.upload_index}:sequence-{self.sequence_index}"
            )
        else:
            self.uid = str(self.uid)
        if self.epoch_mask is not None:
            self.epoch_mask = np.asarray(self.epoch_mask, dtype=bool).reshape(-1)
            if self.epoch_mask.shape != self.pseudo_labels.shape:
                raise ValueError("epoch_mask must match pseudo_labels")
        if self.clean_probabilities is not None:
            self.clean_probabilities = np.asarray(
                self.clean_probabilities, dtype=np.float64
            ).reshape(-1)
            if self.clean_probabilities.shape != self.pseudo_labels.shape:
                raise ValueError("clean_probabilities must match pseudo_labels")

    @property
    def retained_epochs(self) -> int:
        if self.epoch_mask is None:
            return int(self.pseudo_labels.size)
        return int(self.epoch_mask.sum())

    def serializable(self) -> dict[str, Any]:
        return {
            "data_path": str(self.data_path),
            "pseudo_labels": self.pseudo_labels.astype(int).tolist(),
            "task": self.task,
            "subject": self.subject,
            "sequence_index": self.sequence_index,
            "poisoned": bool(self.poisoned),
            "repeated_upload": bool(self.repeated_upload),
            "replay_count": self.replay_count,
            "upload_index": self.upload_index,
            "uid": self.uid,
            "epoch_mask": (
                None if self.epoch_mask is None else self.epoch_mask.astype(bool).tolist()
            ),
            "clean_probabilities": (
                None
                if self.clean_probabilities is None
                else self.clean_probabilities.astype(float).tolist()
            ),
        }


class ReservoirReplayMemory:
    """Fixed-capacity sequence reservoir with an independent deterministic RNG."""

    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.records: list[ReplayRecord] = []
        self.total_seen = 0
        self.rng = np.random.default_rng(seed)
        self.total_replay_draws = 0
        self.poisoned_replay_draws = 0

    def __len__(self) -> int:
        return len(self.records)

    def add(self, incoming: Sequence[ReplayRecord]) -> dict[str, int]:
        inserted = 0
        replaced = 0
        discarded = 0
        for record in incoming:
            self.total_seen += 1
            if len(self.records) < self.capacity:
                self.records.append(record)
                inserted += 1
                continue
            location = int(self.rng.integers(0, self.total_seen))
            if location < self.capacity:
                self.records[location] = record
                replaced += 1
            else:
                discarded += 1
        return {
            "candidates": len(incoming),
            "inserted": inserted,
            "replaced": replaced,
            "discarded": discarded,
            "total_seen": self.total_seen,
            "size": len(self.records),
        }

    def sample(self, count: int) -> list[ReplayRecord]:
        if not self.records or count <= 0:
            return []
        replace = len(self.records) < count
        indices = self.rng.choice(len(self.records), int(count), replace=replace)
        selected = [self.records[int(index)] for index in indices]
        for record in selected:
            record.replay_count += 1
        poisoned = sum(int(record.poisoned) for record in selected)
        self.total_replay_draws += len(selected)
        self.poisoned_replay_draws += poisoned
        return selected

    def stats(self) -> dict[str, Any]:
        poisoned = sum(int(record.poisoned) for record in self.records)
        repeated = sum(int(record.repeated_upload) for record in self.records)
        retained_epochs = sum(record.retained_epochs for record in self.records)
        total_epochs = sum(record.pseudo_labels.size for record in self.records)
        return {
            "capacity": self.capacity,
            "size": len(self.records),
            "total_seen": self.total_seen,
            "unique_paths": len({str(record.data_path) for record in self.records}),
            "unique_occurrences": len({record.uid for record in self.records}),
            "poisoned_records": poisoned,
            "poisoned_fraction": poisoned / max(len(self.records), 1),
            "repeated_upload_records": repeated,
            "repeated_upload_fraction": repeated / max(len(self.records), 1),
            "retained_epochs": retained_epochs,
            "total_epochs": total_epochs,
            "retained_epoch_fraction": retained_epochs / max(total_epochs, 1),
            "total_replay_draws": self.total_replay_draws,
            "poisoned_replay_draws": self.poisoned_replay_draws,
            "poisoned_replay_fraction": (
                self.poisoned_replay_draws / max(self.total_replay_draws, 1)
            ),
        }

    def serializable_records(self) -> list[dict[str, Any]]:
        return [record.serializable() for record in self.records]


def load_replay_batch(
    records: Sequence[ReplayRecord],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load replay signals and encode rejected epochs with ignore index -100."""

    arrays = [
        torch.from_numpy(np.load(record.data_path).astype(np.float32))
        for record in records
    ]
    values = torch.stack(arrays)
    label_rows = []
    for record in records:
        labels = torch.from_numpy(record.pseudo_labels.astype(np.int64).copy())
        if record.epoch_mask is not None:
            mask = torch.from_numpy(record.epoch_mask.astype(bool, copy=False))
            labels[~mask] = -100
        label_rows.append(labels)
    labels = torch.stack(label_rows)
    return values[:, :, :2, :], values[:, :, 2:, :], labels


class _SignalSequenceDataset(Dataset):
    def __init__(self, data_paths: Sequence[Path]):
        self.data_paths = [Path(path) for path in data_paths]

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.from_numpy(
            np.load(self.data_paths[index]).astype(np.float32)
        )
        return values[:, :2, :], values[:, 2:, :]


def _block_outputs(
    blocks,
    eog: torch.Tensor,
    eeg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length = eeg.shape[:2]
    eog_flat = eog.reshape(-1, eog.shape[-2], eog.shape[-1])
    eeg_flat = eeg.reshape(-1, eeg.shape[-2], eeg.shape[-1])
    encoded = blocks[1](blocks[0](eeg_flat, eog_flat))
    encoded = encoded.reshape(batch, sequence_length, -1)
    mlp_features = blocks[2].sleep_stage_mlp(encoded)
    logits = blocks[2].sleep_stage_classifier(mlp_features).permute(0, 2, 1)
    return logits, encoded, mlp_features


@torch.no_grad()
def collect_epoch_outputs(
    blocks,
    data_paths: Sequence[Path],
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect logits, transformer epoch features, and 128-D sequence features."""

    if not data_paths:
        raise ValueError("collect_epoch_outputs requires at least one signal path")
    loader = DataLoader(
        _SignalSequenceDataset(data_paths),
        batch_size=int(getattr(args, "batch", 1)),
        shuffle=False,
        num_workers=int(getattr(args, "num_worker", 0)),
    )
    modes = [block.training for block in blocks]
    for block in blocks:
        block.eval()
    logits_rows: list[torch.Tensor] = []
    epoch_rows: list[torch.Tensor] = []
    sequence_rows: list[torch.Tensor] = []
    try:
        for eog, eeg in loader:
            eog = eog.to(args.device)
            eeg = eeg.to(args.device)
            logits, encoded, mlp_features = _block_outputs(blocks, eog, eeg)
            logits_rows.append(logits.detach().cpu())
            epoch_rows.append(encoded.detach().cpu())
            sequence_rows.append(mlp_features.mean(dim=1).detach().cpu())
    finally:
        for block, mode in zip(blocks, modes):
            block.train(mode)
    return (
        torch.cat(logits_rows, dim=0),
        torch.cat(epoch_rows, dim=0),
        torch.cat(sequence_rows, dim=0),
    )


def apply_spr_filter(
    records: Sequence[ReplayRecord],
    epoch_embeddings: np.ndarray | torch.Tensor,
    args,
    seed: int,
) -> tuple[list[ReplayRecord], dict[str, Any]]:
    """Apply SPR Bernoulli admission to every pseudo-labeled epoch candidate."""

    records = list(records)
    embeddings = _as_numpy(epoch_embeddings, np.float32)
    if embeddings.ndim != 3 or embeddings.shape[0] != len(records):
        raise ValueError("expected epoch_embeddings [N,S,D] aligned with records")
    if records and any(
        record.pseudo_labels.size != embeddings.shape[1] for record in records
    ):
        raise ValueError("record label lengths must match epoch_embeddings")
    if not records:
        return [], {
            "candidate_sequences": 0,
            "accepted_sequences": 0,
            "dropped_sequences": 0,
            "candidate_epochs": 0,
            "retained_epochs": 0,
            "retained_epoch_fraction": 0.0,
            "mean_clean_probability": 0.0,
        }

    flat_features = embeddings.reshape(-1, embeddings.shape[-1])
    flat_labels = np.concatenate([record.pseudo_labels for record in records])
    clean_probabilities = spr_eeg.self_centered_clean_probabilities(
        flat_features,
        flat_labels,
        ensembles=int(getattr(args, "spr_ensembles", 5)),
        bmm_iters=int(getattr(args, "spr_bmm_iters", 10)),
        seed=int(seed),
    ).reshape(len(records), embeddings.shape[1])
    rng = np.random.default_rng(seed)
    accepted = clean_probabilities > rng.random(clean_probabilities.shape)

    retained: list[ReplayRecord] = []
    for index, record in enumerate(records):
        record.clean_probabilities = clean_probabilities[index].astype(
            np.float64, copy=True
        )
        record.epoch_mask = accepted[index].astype(bool, copy=True)
        if record.epoch_mask.any():
            retained.append(record)
    retained_epochs = int(accepted.sum())
    candidate_epochs = int(accepted.size)
    stats = {
        "candidate_sequences": len(records),
        "accepted_sequences": len(retained),
        "dropped_sequences": len(records) - len(retained),
        "candidate_epochs": candidate_epochs,
        "retained_epochs": retained_epochs,
        "retained_epoch_fraction": retained_epochs / max(candidate_epochs, 1),
        "mean_clean_probability": float(clean_probabilities.mean()),
    }
    return retained, stats


def _low_component_probability(
    values: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.float32), []
    finite = np.isfinite(values)
    result = np.ones(values.shape, dtype=np.float32)
    valid = values[finite]
    if valid.size < 8 or float(valid.std()) < 1e-8:
        mean = float(valid.mean()) if valid.size else 0.0
        return result, [mean, mean]
    low, high = float(valid.min()), float(valid.max())
    normalized = (valid - low) / max(high - low, 1e-12)
    try:
        gmm = GaussianMixture(
            n_components=2,
            max_iter=50,
            reg_covar=1e-5,
            random_state=int(seed),
        ).fit(normalized[:, None])
        means = gmm.means_.reshape(-1)
        low_component = int(np.argmin(means))
        result[finite] = gmm.predict_proba(normalized[:, None])[
            :, low_component
        ].astype(np.float32)
        original_means = sorted(float(low + value * (high - low)) for value in means)
    except (FloatingPointError, ValueError):
        mean = float(valid.mean())
        original_means = [mean, mean]
    return result, original_means


def _threshold_pair(thresholds: Any) -> tuple[float, float]:
    if thresholds is None:
        return 0.5, 0.5
    if isinstance(thresholds, Mapping):
        clean = thresholds.get(
            "clean", thresholds.get("clean_threshold", thresholds.get("c", 0.5))
        )
        uncertainty = thresholds.get(
            "uncertainty",
            thresholds.get("uncertainty_threshold", thresholds.get("r", 0.5)),
        )
        return float(clean), float(uncertainty)
    if isinstance(thresholds, Sequence) and not isinstance(thresholds, (str, bytes)):
        if len(thresholds) != 2:
            raise ValueError("thresholds must contain clean and uncertainty values")
        return float(thresholds[0]), float(thresholds[1])
    clean = getattr(
        thresholds,
        "puridiver_clean_threshold",
        getattr(thresholds, "clean_threshold", 0.5),
    )
    uncertainty = getattr(
        thresholds,
        "puridiver_uncertainty_threshold",
        getattr(thresholds, "uncertainty_threshold", 0.5),
    )
    return float(clean), float(uncertainty)


@dataclass(frozen=True)
class CRUState:
    """Frozen clean/relabel/unlabeled assignment for one replay batch."""

    clean_probability: torch.Tensor
    low_uncertainty_probability: torch.Tensor
    snapshot_probabilities: torch.Tensor
    clean_mask: torch.Tensor
    relabel_mask: torch.Tensor
    unlabeled_mask: torch.Tensor
    diagnostics: dict[str, Any]

    @property
    def c_mask(self) -> torch.Tensor:
        return self.clean_mask

    @property
    def r_mask(self) -> torch.Tensor:
        return self.relabel_mask

    @property
    def u_mask(self) -> torch.Tensor:
        return self.unlabeled_mask

    @property
    def model_probability(self) -> torch.Tensor:
        return self.snapshot_probabilities


def _class_last_logits(logits: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if logits.ndim == observed.ndim + 1 and logits.shape[:-1] == observed.shape:
        return logits
    if (
        logits.ndim == observed.ndim + 1
        and logits.shape[0] == observed.shape[0]
        and logits.shape[2:] == observed.shape[1:]
    ):
        order = [0, *range(2, logits.ndim), 1]
        return logits.permute(*order)
    raise ValueError("logits must be [M,C], [N,C,S], or class-last equivalents")


def build_cru_state(
    logits: torch.Tensor | np.ndarray,
    observed: torch.Tensor | np.ndarray,
    seed: int,
    thresholds: Any = None,
) -> CRUState:
    """Fit loss GMM, then uncertainty GMM only within its noisy subset."""

    logits_tensor = torch.as_tensor(logits, dtype=torch.float32).detach().cpu()
    observed_tensor = torch.as_tensor(observed, dtype=torch.long).detach().cpu()
    class_last = _class_last_logits(logits_tensor, observed_tensor)
    classes = class_last.shape[-1]
    flat_logits = class_last.reshape(-1, classes)
    flat_observed = observed_tensor.reshape(-1)
    if flat_logits.shape[0] != flat_observed.numel() or flat_observed.numel() == 0:
        raise ValueError("logits and observed labels must be non-empty and aligned")
    valid = flat_observed.ne(-100)
    if not bool(valid.any()):
        raise ValueError("observed labels contain no valid classes")
    invalid = valid & ((flat_observed < 0) | (flat_observed >= classes))
    if bool(invalid.any()):
        raise ValueError("observed labels contain an invalid class")

    probabilities = flat_logits.softmax(dim=1)
    valid_logits = flat_logits[valid]
    valid_observed = flat_observed[valid]
    losses = F.cross_entropy(valid_logits, valid_observed, reduction="none").numpy()
    valid_clean_probability, loss_means = _low_component_probability(losses, seed)
    clean_threshold, uncertainty_threshold = _threshold_pair(thresholds)
    valid_clean = valid_clean_probability >= clean_threshold
    noisy_indices = np.flatnonzero(~valid_clean)

    valid_low_uncertainty = np.zeros(valid_observed.numel(), dtype=np.float32)
    uncertainty_means: list[float] | None = None
    if noisy_indices.size:
        uncertainty = 1.0 - probabilities[valid].max(dim=1).values.numpy()
        noisy_probability, uncertainty_means = _low_component_probability(
            uncertainty[noisy_indices], seed + 1
        )
        valid_low_uncertainty[noisy_indices] = noisy_probability
    valid_relabel = (~valid_clean) & (
        valid_low_uncertainty >= uncertainty_threshold
    )
    valid_unlabeled = (~valid_clean) & (~valid_relabel)

    clean_probability = np.zeros(flat_observed.numel(), dtype=np.float32)
    low_uncertainty = np.zeros(flat_observed.numel(), dtype=np.float32)
    clean = np.zeros(flat_observed.numel(), dtype=bool)
    relabel = np.zeros(flat_observed.numel(), dtype=bool)
    unlabeled = np.zeros(flat_observed.numel(), dtype=bool)
    valid_indices = valid.numpy()
    clean_probability[valid_indices] = valid_clean_probability
    low_uncertainty[valid_indices] = valid_low_uncertainty
    clean[valid_indices] = valid_clean
    relabel[valid_indices] = valid_relabel
    unlabeled[valid_indices] = valid_unlabeled

    shape = observed_tensor.shape
    diagnostics = {
        "loss_gmm_means": loss_means,
        "uncertainty_gmm_means": uncertainty_means,
        "clean_count": int(clean.sum()),
        "relabel_count": int(relabel.sum()),
        "unlabeled_count": int(unlabeled.sum()),
        "ignored_count": int((~valid.numpy()).sum()),
        "clean_fraction": float(valid_clean.mean()),
        "relabel_fraction": float(valid_relabel.mean()),
        "unlabeled_fraction": float(valid_unlabeled.mean()),
    }
    return CRUState(
        clean_probability=torch.from_numpy(clean_probability.reshape(shape)),
        low_uncertainty_probability=torch.from_numpy(low_uncertainty.reshape(shape)),
        snapshot_probabilities=probabilities.reshape(*shape, classes),
        clean_mask=torch.from_numpy(clean.reshape(shape)),
        relabel_mask=torch.from_numpy(relabel.reshape(shape)),
        unlabeled_mask=torch.from_numpy(unlabeled.reshape(shape)),
        diagnostics=diagnostics,
    )


def _arg(args, *names: str, default: float) -> float:
    for name in names:
        if hasattr(args, name):
            return float(getattr(args, name))
    return float(default)


def _strong_augment(
    signal: torch.Tensor,
    noise_ratio: float,
    scale_ratio: float,
    mask_fraction: float,
) -> torch.Tensor:
    std = signal.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = 1.0 + (
        2.0 * torch.rand((*signal.shape[:-1], 1), device=signal.device) - 1.0
    ) * scale_ratio
    augmented = signal * scale + torch.randn_like(signal) * std * noise_ratio
    if mask_fraction > 0.0:
        width = min(signal.shape[-1], max(1, round(signal.shape[-1] * mask_fraction)))
        starts = torch.randint(
            0,
            max(signal.shape[-1] - width + 1, 1),
            signal.shape[:-1],
            device=signal.device,
        )
        positions = torch.arange(signal.shape[-1], device=signal.device)
        mask = (positions >= starts.unsqueeze(-1)) & (
            positions < starts.unsqueeze(-1) + width
        )
        augmented = augmented.masked_fill(mask, 0.0)
    return augmented


def puridiver_branch_loss(
    blocks,
    eog: torch.Tensor,
    eeg: torch.Tensor,
    observed: torch.Tensor,
    state: CRUState,
    args,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute differentiable PuriDivER C/R/U branch loss."""

    observed = observed.to(eeg.device).long()
    logits, _encoded, _mlp = _block_outputs(blocks, eog, eeg)
    class_last = logits.permute(0, 2, 1)
    classes = class_last.shape[-1]
    flat_logits = class_last.reshape(-1, classes)
    flat_observed = observed.reshape(-1)

    clean = state.clean_mask.to(eeg.device).reshape(-1).bool()
    relabel = state.relabel_mask.to(eeg.device).reshape(-1).bool()
    unlabeled = state.unlabeled_mask.to(eeg.device).reshape(-1).bool()
    valid = flat_observed.ne(-100)
    clean &= valid
    relabel &= valid
    unlabeled &= valid
    if clean.numel() != flat_observed.numel():
        raise ValueError("CRU state shape must match observed labels")

    numerator = flat_logits.sum() * 0.0
    clean_sum = flat_logits.new_zeros(())
    relabel_sum = flat_logits.new_zeros(())
    consistency_sum = flat_logits.new_zeros(())
    if bool(clean.any()):
        clean_sum = F.cross_entropy(
            flat_logits[clean], flat_observed[clean], reduction="sum"
        )
        numerator = numerator + clean_sum
    if bool(relabel.any()):
        q = state.low_uncertainty_probability.to(eeg.device).reshape(-1)[
            relabel
        ].unsqueeze(1)
        snapshot = state.snapshot_probabilities.to(eeg.device).reshape(-1, classes)[
            relabel
        ]
        one_hot = F.one_hot(flat_observed[relabel], classes).to(flat_logits.dtype)
        soft_target = q * snapshot + (1.0 - q) * one_hot
        relabel_sum = -(
            soft_target * F.log_softmax(flat_logits[relabel], dim=1)
        ).sum()
        numerator = numerator + relabel_sum
    if bool(unlabeled.any()):
        noise = _arg(
            args, "puridiver_strong_noise", "strong_noise", default=0.01
        )
        scale = _arg(
            args, "puridiver_strong_scale", "strong_scale", default=0.08
        )
        mask_fraction = _arg(
            args,
            "puridiver_strong_mask_fraction",
            "strong_mask_fraction",
            default=0.0,
        )
        strong_eog = _strong_augment(eog, noise, scale, mask_fraction)
        strong_eeg = _strong_augment(eeg, noise, scale, mask_fraction)
        strong_logits, _strong_encoded, _strong_mlp = _block_outputs(
            blocks, strong_eog, strong_eeg
        )
        weak_probability = class_last.softmax(dim=-1).detach().reshape(-1, classes)
        strong_probability = strong_logits.permute(0, 2, 1).softmax(dim=-1).reshape(
            -1, classes
        )
        consistency_sum = F.mse_loss(
            strong_probability[unlabeled],
            weak_probability[unlabeled],
            reduction="sum",
        )
        weight = _arg(
            args,
            "puridiver_consistency_weight",
            "consistency_weight",
            default=1.0,
        )
        numerator = numerator + weight * consistency_sum

    branch_count = int(clean.sum() + relabel.sum() + unlabeled.sum())
    loss = numerator / max(branch_count, 1)
    stats = {
        "clean_count": int(clean.sum()),
        "relabel_count": int(relabel.sum()),
        "unlabeled_count": int(unlabeled.sum()),
        "clean_loss_sum": float(clean_sum.detach().cpu()),
        "relabel_loss_sum": float(relabel_sum.detach().cpu()),
        "consistency_loss_sum": float(consistency_sum.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, stats


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - values.mean()) / std


@torch.no_grad()
def _sequence_scores(
    records: Sequence[ReplayRecord],
    blocks,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    losses: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    classes: list[torch.Tensor] = []
    batch_size = max(1, int(getattr(args, "batch", 1)))
    modes = [block.training for block in blocks]
    for block in blocks:
        block.eval()
    try:
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            eog, eeg, observed = load_replay_batch(batch_records)
            eog = eog.to(args.device)
            eeg = eeg.to(args.device)
            observed = observed.to(args.device)
            logits, _encoded, mlp_features = _block_outputs(blocks, eog, eeg)
            class_last = logits.permute(0, 2, 1)
            epoch_loss = F.cross_entropy(
                class_last.reshape(-1, class_last.shape[-1]),
                observed.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).reshape(observed.shape)
            valid = observed.ne(-100)
            losses.append(
                (epoch_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            )
            features.append(mlp_features.mean(dim=1))
            dominant = []
            for row in observed:
                kept = row[row.ge(0)]
                dominant.append(
                    torch.bincount(kept, minlength=class_last.shape[-1]).argmax()
                    if kept.numel()
                    else torch.zeros((), dtype=torch.long, device=row.device)
                )
            classes.append(torch.stack(dominant))
    finally:
        for block, mode in zip(blocks, modes):
            block.train(mode)
    return (
        torch.cat(losses).cpu().numpy().astype(np.float64),
        torch.cat(features).cpu().numpy().astype(np.float64),
        torch.cat(classes).cpu().numpy().astype(np.int64),
    )


class PuriDivERSequenceMemory:
    """PuriDivER purity/diversity pruning over up to 1000 EEG sequences."""

    def __init__(self, capacity: int = 1000, seed: int = 0):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.records: list[ReplayRecord] = []
        self.total_seen = 0
        self.rng = np.random.default_rng(seed)
        self.total_replay_draws = 0
        self.poisoned_replay_draws = 0
        self.total_pruned = 0

    def __len__(self) -> int:
        return len(self.records)

    def sample(self, count: int) -> list[ReplayRecord]:
        if not self.records or count <= 0:
            return []
        indices = self.rng.choice(
            len(self.records), int(count), replace=len(self.records) < count
        )
        selected = [self.records[int(index)] for index in indices]
        for record in selected:
            record.replay_count += 1
        poisoned = sum(int(record.poisoned) for record in selected)
        self.total_replay_draws += len(selected)
        self.poisoned_replay_draws += poisoned
        return selected

    def add(
        self,
        incoming: Sequence[ReplayRecord],
        blocks,
        args,
        diversity_coefficient: float,
    ) -> dict[str, Any]:
        alpha = float(diversity_coefficient)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("diversity_coefficient must be in [0, 1]")
        incoming = list(incoming)
        previous_uids = {record.uid for record in self.records}
        candidates = [*self.records, *incoming]
        self.total_seen += len(incoming)
        if len(candidates) <= self.capacity:
            self.records = candidates
            return {
                "candidates": len(candidates),
                "incoming": len(incoming),
                "inserted": len(incoming),
                "removed": 0,
                "discarded": 0,
                "size": len(self.records),
                "total_seen": self.total_seen,
                "retained_epochs": sum(record.retained_epochs for record in self.records),
            }

        losses, features, dominant_classes = _sequence_scores(candidates, blocks, args)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized_features = features / np.maximum(norms, 1e-12)
        active = np.ones(len(candidates), dtype=bool)
        remove_count = len(candidates) - self.capacity
        removed_scores: list[float] = []

        for _ in range(remove_count):
            active_classes = dominant_classes[active]
            unique_classes, counts = np.unique(active_classes, return_counts=True)
            largest = unique_classes[counts == counts.max()]
            class_id = int(self.rng.choice(largest))
            class_indices = np.flatnonzero(active & (dominant_classes == class_id))
            class_features = normalized_features[class_indices]
            centroid = class_features.mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            redundancy = class_features @ centroid
            scores = (1.0 - alpha) * _zscore(losses[class_indices]) + alpha * _zscore(
                redundancy
            )
            maximum = np.flatnonzero(np.isclose(scores, scores.max()))
            remove_position = int(self.rng.choice(maximum))
            remove_index = int(class_indices[remove_position])
            active[remove_index] = False
            removed_scores.append(float(scores[remove_position]))

        self.records = [
            record for index, record in enumerate(candidates) if bool(active[index])
        ]
        self.total_pruned += remove_count
        retained_uids = {record.uid for record in self.records}
        inserted = sum(record.uid in retained_uids for record in incoming)
        discarded = len(incoming) - inserted
        old_removed = sum(uid not in retained_uids for uid in previous_uids)
        class_counts = {
            str(int(class_id)): int((dominant_classes[active] == class_id).sum())
            for class_id in np.unique(dominant_classes[active])
        }
        return {
            "candidates": len(candidates),
            "incoming": len(incoming),
            "inserted": inserted,
            "removed": remove_count,
            "old_removed": old_removed,
            "discarded": discarded,
            "size": len(self.records),
            "total_seen": self.total_seen,
            "class_counts": class_counts,
            "mean_removed_score": float(np.mean(removed_scores)),
            "retained_epochs": sum(record.retained_epochs for record in self.records),
        }

    def stats(self) -> dict[str, Any]:
        poisoned = sum(int(record.poisoned) for record in self.records)
        repeated = sum(int(record.repeated_upload) for record in self.records)
        retained_epochs = sum(record.retained_epochs for record in self.records)
        total_epochs = sum(record.pseudo_labels.size for record in self.records)
        dominant_classes = [
            int(
                np.bincount(
                    record.pseudo_labels[
                        np.ones(record.pseudo_labels.shape, dtype=bool)
                        if record.epoch_mask is None
                        else record.epoch_mask
                    ],
                    minlength=max(int(record.pseudo_labels.max()) + 1, 1),
                ).argmax()
            )
            for record in self.records
            if record.retained_epochs
        ]
        return {
            "capacity": self.capacity,
            "size": len(self.records),
            "total_seen": self.total_seen,
            "unique_paths": len({str(record.data_path) for record in self.records}),
            "unique_occurrences": len({record.uid for record in self.records}),
            "poisoned_records": poisoned,
            "poisoned_fraction": poisoned / max(len(self.records), 1),
            "repeated_upload_records": repeated,
            "repeated_upload_fraction": repeated / max(len(self.records), 1),
            "retained_epochs": retained_epochs,
            "total_epochs": total_epochs,
            "retained_epoch_fraction": retained_epochs / max(total_epochs, 1),
            "class_counts": {
                str(class_id): dominant_classes.count(class_id)
                for class_id in sorted(set(dominant_classes))
            },
            "total_pruned": self.total_pruned,
            "total_replay_draws": self.total_replay_draws,
            "poisoned_replay_draws": self.poisoned_replay_draws,
            "poisoned_replay_fraction": (
                self.poisoned_replay_draws / max(self.total_replay_draws, 1)
            ),
        }

    def serializable_records(self) -> list[dict[str, Any]]:
        return [record.serializable() for record in self.records]


def build_memory_records(
    data_paths: Sequence[Path],
    pseudo_labels: Sequence[np.ndarray],
    *,
    task_index: int,
    subject: int,
    original_count: int,
    poisoned_paths: set[str],
    sequence_indices: Sequence[int] | None = None,
    epoch_masks: Sequence[np.ndarray | None] | None = None,
    clean_probabilities: Sequence[np.ndarray | None] | None = None,
) -> list[ReplayRecord]:
    """Build one independent record for every uploaded path occurrence."""

    count = len(data_paths)
    if len(pseudo_labels) != count:
        raise ValueError("Replay admission data/label count mismatch")
    if epoch_masks is not None and len(epoch_masks) != count:
        raise ValueError("Replay admission data/mask count mismatch")
    if clean_probabilities is not None and len(clean_probabilities) != count:
        raise ValueError("Replay admission data/probability count mismatch")
    if sequence_indices is not None and len(sequence_indices) != count:
        raise ValueError("Replay admission data/sequence-index count mismatch")

    records: list[ReplayRecord] = []
    for upload_index, (path, labels) in enumerate(zip(data_paths, pseudo_labels)):
        path = Path(path)
        sequence_index = (
            int(path.stem)
            if sequence_indices is None
            else int(sequence_indices[upload_index])
        )
        records.append(
            ReplayRecord(
                data_path=path,
                pseudo_labels=np.asarray(labels, dtype=np.int64),
                task=int(task_index),
                subject=int(subject),
                sequence_index=sequence_index,
                poisoned=str(path) in poisoned_paths,
                repeated_upload=upload_index >= original_count,
                upload_index=upload_index,
                uid=(
                    f"task-{int(task_index)}:subject-{int(subject)}:"
                    f"upload-{upload_index}:sequence-{sequence_index}"
                ),
                epoch_mask=(None if epoch_masks is None else epoch_masks[upload_index]),
                clean_probabilities=(
                    None
                    if clean_probabilities is None
                    else clean_probabilities[upload_index]
                ),
            )
        )
    return records


__all__ = [
    "CRUState",
    "PuriDivERSequenceMemory",
    "ReplayRecord",
    "ReservoirReplayMemory",
    "apply_spr_filter",
    "build_cru_state",
    "build_memory_records",
    "collect_epoch_outputs",
    "load_replay_batch",
    "puridiver_branch_loss",
]
