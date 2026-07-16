"""Standalone PuriDivER-style continual learning on labeled ISRUC streams.

This entry point deliberately does not use BrainUICL's teacher, pseudo labels,
CPC, CEA, pretrained checkpoints, or continual-learning objective.  It uses a
small randomly initialized EEG classifier and maps each 30-second sleep epoch
to one PuriDivER sample.  Subjects are continual tasks, with disjoint sequence
blocks for online adaptation and held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
NUM_CLASSES = len(CLASS_NAMES)


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discover_subjects(data_root: Path) -> list[int]:
    subjects = []
    for path in data_root.iterdir():
        if path.is_dir() and path.name.isdigit() and (path / "data" / "0.npy").exists():
            subjects.append(int(path.name))
    if not subjects:
        raise RuntimeError(f"No processed ISRUC subjects found under {data_root}")
    return sorted(subjects)


def reference_subject_split(subjects: list[int], seed: int) -> dict[str, list[int]]:
    """Reproduce the existing ISRUC subject split solely for comparability."""
    rng = np.random.RandomState(seed)
    subject_ids = list(subjects)
    old_count = max(1, int(len(subject_ids) * 0.2))
    new_count = max(1, int(len(subject_ids) * 0.5))
    new_count = min(new_count, len(subject_ids) - old_count - 2)
    old = rng.choice(subject_ids, old_count, replace=False).tolist()
    remaining = sorted(set(subject_ids) - set(old))
    new = rng.choice(remaining, new_count, replace=False).tolist()
    train_val = sorted(set(subject_ids) - set(old) - set(new))
    train_count = min(max(1, int(len(train_val) * 0.8)), len(train_val) - 1)
    train = rng.choice(train_val, train_count, replace=False).tolist()
    val = sorted(set(train_val) - set(train))
    return {
        "train": sorted(int(x) for x in train),
        "val": sorted(int(x) for x in val),
        "old_generalization": sorted(int(x) for x in old),
        "new_order": [int(x) for x in new],
    }


def _numeric_npy_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.npy"), key=lambda path: int(path.stem))


def split_subject_paths(data_root: Path, subject: int, train_fraction: float) -> tuple[list[Path], list[Path]]:
    paths = _numeric_npy_paths(data_root / str(subject) / "data")
    if len(paths) < 2:
        raise RuntimeError(f"Subject {subject} needs at least two sequences, found {len(paths)}")
    split = int(math.floor(len(paths) * train_fraction))
    split = min(max(split, 1), len(paths) - 1)
    return paths[:split], paths[split:]


def normalize_epochs(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    mean = values.mean(axis=-1, keepdims=True)
    std = values.std(axis=-1, keepdims=True)
    normalized = (values - mean) / np.maximum(std, 1e-7)
    return np.clip(normalized, -10.0, 10.0)


def symmetric_noise(labels: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(labels, dtype=np.int64).copy()
    if rate <= 0:
        return observed, np.zeros_like(observed, dtype=bool)
    rng = np.random.default_rng(seed)
    mask = rng.random(observed.shape) < rate
    offsets = rng.integers(1, NUM_CLASSES, size=observed.shape)
    observed[mask] = (observed[mask] + offsets[mask]) % NUM_CLASSES
    return observed, mask


@dataclass
class EpochPool:
    x: torch.Tensor
    observed_y: torch.Tensor
    true_y: torch.Tensor
    subject_y: torch.Tensor

    def __len__(self) -> int:
        return int(self.observed_y.numel())

    def subset(self, indices: torch.Tensor | np.ndarray | list[int]) -> "EpochPool":
        indices = torch.as_tensor(indices, dtype=torch.long)
        return EpochPool(
            self.x[indices],
            self.observed_y[indices],
            self.true_y[indices],
            self.subject_y[indices],
        )

    @staticmethod
    def empty() -> "EpochPool":
        return EpochPool(
            torch.empty((0, 8, 3000), dtype=torch.float16),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
        )

    @staticmethod
    def concatenate(pools: list["EpochPool"]) -> "EpochPool":
        pools = [pool for pool in pools if len(pool)]
        if not pools:
            return EpochPool.empty()
        return EpochPool(
            torch.cat([pool.x for pool in pools], dim=0),
            torch.cat([pool.observed_y for pool in pools], dim=0),
            torch.cat([pool.true_y for pool in pools], dim=0),
            torch.cat([pool.subject_y for pool in pools], dim=0),
        )


def diagnostic_label_purity(pool: EpochPool) -> float | None:
    """Return observed-label purity when hidden annotations are available."""

    valid = pool.true_y.ge(0)
    if not bool(valid.any()):
        return None
    return float(pool.observed_y[valid].eq(pool.true_y[valid]).float().mean())


class PoolDataset(Dataset):
    def __init__(self, pool: EpochPool):
        self.pool = pool

    def __len__(self) -> int:
        return len(self.pool)

    def __getitem__(self, index: int):
        return (
            self.pool.x[index].float(),
            self.pool.observed_y[index],
            self.pool.true_y[index],
            index,
        )


def load_epoch_pool(
    data_root: Path,
    subject: int,
    paths: list[Path],
    noise_rate: float,
    noise_seed: int,
) -> tuple[EpochPool, dict]:
    rows, labels = [], []
    label_root = data_root / str(subject) / "label"
    for path in paths:
        values = normalize_epochs(np.load(path))
        true_labels = np.load(label_root / path.name).astype(np.int64)
        if values.shape[0] != true_labels.shape[0]:
            raise ValueError(f"Data/label length mismatch for {path}")
        rows.append(torch.from_numpy(values).to(torch.float16))
        labels.append(true_labels)
    x = torch.cat(rows, dim=0)
    true_y_np = np.concatenate(labels)
    observed_np, noisy_mask = symmetric_noise(true_y_np, noise_rate, noise_seed)
    pool = EpochPool(
        x=x,
        observed_y=torch.from_numpy(observed_np),
        true_y=torch.from_numpy(true_y_np),
        subject_y=torch.full((len(true_y_np),), int(subject), dtype=torch.long),
    )
    return pool, {
        "epochs": len(pool),
        "sequences": len(paths),
        "noise_count": int(noisy_mask.sum()),
        "realized_noise_rate": float(noisy_mask.mean()),
        "class_counts": np.bincount(true_y_np, minlength=NUM_CLASSES).astype(int).tolist(),
    }


class CompactEEGClassifier(nn.Module):
    """Plain single-epoch EEG backbone; no BrainUICL method components."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=51, stride=10, padding=25, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(3, 2),
            nn.Conv1d(32, 64, kernel_size=15, stride=3, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, NUM_CLASSES, bias=False)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.encoder(x).squeeze(-1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


@torch.no_grad()
def infer_pool(
    model: nn.Module,
    pool: EpochPool,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(PoolDataset(pool), batch_size=batch_size, shuffle=False, num_workers=0)
    logits, features = [], []
    model.eval()
    for x, _observed, _true, _index in loader:
        output, encoded = model(x.to(device), return_features=True)
        logits.append(output.cpu())
        features.append(encoded.cpu())
    return torch.cat(logits), torch.cat(features)


def fit_low_component_probability(values: np.ndarray, seed: int) -> tuple[np.ndarray, list[float]]:
    """Fit a 2-GMM and return posterior of its lower-mean component."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 8 or not np.isfinite(values).all() or float(values.std()) < 1e-8:
        return np.ones(values.shape, dtype=np.float32), [float(values.mean()), float(values.mean())]
    gmm = GaussianMixture(
        n_components=2,
        max_iter=50,
        tol=1e-3,
        reg_covar=5e-4,
        random_state=seed,
    ).fit(values[:, None])
    means = gmm.means_.reshape(-1)
    low_component = int(np.argmin(means))
    posterior = gmm.predict_proba(values[:, None])[:, low_component].astype(np.float32)
    return posterior, sorted(float(value) for value in means)


@dataclass
class PurificationState:
    clean_probability: torch.Tensor
    low_uncertainty_probability: torch.Tensor
    model_probability: torch.Tensor
    clean_mask: torch.Tensor
    relabel_mask: torch.Tensor
    unlabeled_mask: torch.Tensor
    diagnostics: dict


def puridiver_split(
    model: nn.Module,
    memory: EpochPool,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> PurificationState:
    logits, _features = infer_pool(model, memory, device, batch_size)
    probabilities = logits.softmax(dim=1)
    losses = F.cross_entropy(logits, memory.observed_y, reduction="none").numpy()
    loss_range = float(losses.max() - losses.min())
    normalized_losses = (losses - losses.min()) / max(loss_range, 1e-12)
    clean_probability, loss_means = fit_low_component_probability(normalized_losses, seed)
    clean_mask_np = clean_probability >= 0.5
    noisy_indices = np.flatnonzero(~clean_mask_np)

    low_uncertainty_probability = np.zeros(len(memory), dtype=np.float32)
    uncertainty_means: list[float] | None = None
    if noisy_indices.size:
        uncertainty = 1.0 - probabilities.max(dim=1).values.numpy()
        noisy_posterior, uncertainty_means = fit_low_component_probability(
            uncertainty[noisy_indices], seed + 1
        )
        low_uncertainty_probability[noisy_indices] = noisy_posterior

    relabel_mask_np = (~clean_mask_np) & (low_uncertainty_probability >= 0.5)
    unlabeled_mask_np = (~clean_mask_np) & (~relabel_mask_np)
    annotation_available = memory.true_y.ge(0).numpy()
    true_clean = memory.observed_y.eq(memory.true_y).numpy()

    def precision(mask: np.ndarray) -> float | None:
        diagnostic_mask = mask & annotation_available
        return float(true_clean[diagnostic_mask].mean()) if diagnostic_mask.any() else None

    diagnostics = {
        "loss_gmm_means": loss_means,
        "uncertainty_gmm_means": uncertainty_means,
        "clean_count": int(clean_mask_np.sum()),
        "relabel_count": int(relabel_mask_np.sum()),
        "unlabeled_count": int(unlabeled_mask_np.sum()),
        "clean_precision": precision(clean_mask_np),
        "relabel_observed_label_precision": precision(relabel_mask_np),
        "memory_label_purity": diagnostic_label_purity(memory),
    }
    return PurificationState(
        clean_probability=torch.from_numpy(clean_probability),
        low_uncertainty_probability=torch.from_numpy(low_uncertainty_probability),
        model_probability=probabilities,
        clean_mask=torch.from_numpy(clean_mask_np),
        relabel_mask=torch.from_numpy(relabel_mask_np),
        unlabeled_mask=torch.from_numpy(unlabeled_mask_np),
        diagnostics=diagnostics,
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.mean()) / std


class PuriDivERMemory:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.pool = EpochPool.empty()
        self.rng = np.random.default_rng(seed)
        self.reservoir_seen = 0

    def __len__(self) -> int:
        return len(self.pool)

    def sample(self, count: int) -> EpochPool:
        if not len(self.pool) or count <= 0:
            return EpochPool.empty()
        indices = self.rng.choice(len(self.pool), size=count, replace=len(self.pool) < count)
        return self.pool.subset(indices)

    def reservoir_update(self, incoming: EpochPool) -> dict:
        for index in range(len(incoming)):
            self.reservoir_seen += 1
            item = incoming.subset([index])
            if len(self.pool) < self.capacity:
                self.pool = EpochPool.concatenate([self.pool, item])
                continue
            replacement = int(self.rng.integers(0, self.reservoir_seen))
            if replacement < self.capacity:
                self.pool.x[replacement] = item.x[0]
                self.pool.observed_y[replacement] = item.observed_y[0]
                self.pool.true_y[replacement] = item.true_y[0]
                self.pool.subject_y[replacement] = item.subject_y[0]
        return {
            "strategy": "reservoir",
            "candidates": self.reservoir_seen,
            "size": len(self.pool),
            "purity": diagnostic_label_purity(self.pool),
        }

    @torch.no_grad()
    def update(
        self,
        incoming: EpochPool,
        model: nn.Module,
        device: torch.device,
        infer_batch_size: int,
        diversity_coefficient: float,
    ) -> dict:
        candidates = EpochPool.concatenate([self.pool, incoming])
        if len(candidates) <= self.capacity:
            self.pool = candidates
            return {
                "candidates": len(candidates),
                "removed": 0,
                "size": len(self.pool),
                "purity": diagnostic_label_purity(self.pool),
            }

        logits, features = infer_pool(model, candidates, device, infer_batch_size)
        losses = F.cross_entropy(logits, candidates.observed_y, reduction="none").numpy()
        features_np = features.numpy().astype(np.float64)
        labels = candidates.observed_y.numpy()
        classifier_weight = model.classifier.weight.detach().cpu()
        mean_weight = classifier_weight.mean(dim=0)
        relevant_features: dict[int, np.ndarray] = {}
        for class_id in range(NUM_CLASSES):
            relevant = classifier_weight[class_id] > mean_weight
            if not bool(relevant.any()):
                relevant = torch.ones_like(relevant, dtype=torch.bool)
            selected_features = features_np[:, relevant.numpy()]
            norms = np.linalg.norm(selected_features, axis=1, keepdims=True)
            relevant_features[class_id] = selected_features / np.maximum(norms, 1e-12)

        active = np.ones(len(candidates), dtype=bool)
        remove_count = len(candidates) - self.capacity
        removed_scores = []
        for _ in range(remove_count):
            counts = np.bincount(labels[active], minlength=NUM_CLASSES)
            largest_classes = np.flatnonzero(counts == counts.max())
            class_id = int(self.rng.choice(largest_classes))
            class_indices = np.flatnonzero(active & (labels == class_id))
            class_features = relevant_features[class_id][class_indices]
            similarity = class_features @ class_features.mean(axis=0)
            class_scores = (
                (1.0 - diversity_coefficient) * _zscore(losses[class_indices])
                + diversity_coefficient * _zscore(similarity)
            )
            remove_position = int(np.argmax(class_scores))
            remove_index = int(class_indices[remove_position])
            removed_scores.append(float(class_scores[remove_position]))
            active[remove_index] = False

        self.pool = candidates.subset(np.flatnonzero(active))
        kept_scores = []
        for class_id in range(NUM_CLASSES):
            class_indices = np.flatnonzero(active & (labels == class_id))
            if not class_indices.size:
                continue
            class_features = relevant_features[class_id][class_indices]
            similarity = class_features @ class_features.mean(axis=0)
            kept_scores.extend(
                (
                    (1.0 - diversity_coefficient) * _zscore(losses[class_indices])
                    + diversity_coefficient * _zscore(similarity)
                ).tolist()
            )
        return {
            "candidates": len(candidates),
            "removed": remove_count,
            "size": len(self.pool),
            "purity": diagnostic_label_purity(self.pool),
            "class_counts": torch.bincount(self.pool.observed_y, minlength=NUM_CLASSES).tolist(),
            "mean_kept_score": float(np.mean(kept_scores)),
            "mean_removed_score": float(np.mean(removed_scores)),
        }


def make_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )


def _augment(x: torch.Tensor, noise: float, scale: float, mask_fraction: float) -> torch.Tensor:
    scale_values = 1.0 + (2.0 * torch.rand((*x.shape[:-1], 1), device=x.device) - 1.0) * scale
    augmented = x * scale_values + torch.randn_like(x) * noise
    if mask_fraction > 0:
        mask_width = max(1, int(x.shape[-1] * mask_fraction))
        starts = torch.randint(0, x.shape[-1] - mask_width + 1, (x.shape[0],), device=x.device)
        for row, start in enumerate(starts.tolist()):
            augmented[row, :, start : start + mask_width] = 0.0
    return augmented


def online_train_subject(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stream: EpochPool,
    memory: PuriDivERMemory,
    args,
    task_index: int,
) -> dict:
    generator = torch.Generator().manual_seed(args.seed + 1000 * task_index)
    loader = DataLoader(
        PoolDataset(stream),
        batch_size=args.online_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    losses, coefficients, updates = [], [], []
    model.train()
    for x, observed_y, true_y, indices in loader:
        current = EpochPool(
            x=x.to(torch.float16),
            observed_y=observed_y.clone(),
            true_y=true_y.clone(),
            subject_y=stream.subject_y[indices],
        )
        replay = memory.sample(len(current)) if task_index > 1 else EpochPool.empty()
        train_pool = EpochPool.concatenate([current, replay])
        train_x = train_pool.x.float().to(args.device)
        train_y = train_pool.observed_y.to(args.device)
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        coefficient = args.max_diversity_coefficient * min(1.0 / max(loss_value, 1e-8), 1.0)
        if args.method in {"puridiver", "puri_memory"}:
            update = memory.update(
                current,
                model,
                args.device,
                args.infer_batch_size,
                coefficient,
            )
        else:
            update = memory.reservoir_update(current)
        losses.append(loss_value)
        coefficients.append(coefficient)
        updates.append(update)
    return {
        "mini_batches": len(losses),
        "mean_online_loss": float(np.mean(losses)),
        "mean_diversity_coefficient": float(np.mean(coefficients)),
        "last_memory_update": updates[-1],
    }


def standard_replay_epoch(model, optimizer, memory: EpochPool, args, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PoolDataset(memory),
        batch_size=args.replay_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    losses = []
    model.train()
    for x, observed_y, _true_y, _indices in loader:
        logits = model(x.to(args.device))
        loss = F.cross_entropy(logits, observed_y.to(args.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def puridiver_replay_epoch(model, optimizer, memory: EpochPool, state: PurificationState, args, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PoolDataset(memory),
        batch_size=args.replay_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    losses = []
    model.train()
    for x, observed_y, _true_y, indices in loader:
        device = args.device
        x = x.to(device)
        observed_y = observed_y.to(device)
        indices = indices.long()
        clean_mask = state.clean_mask[indices].to(device)
        relabel_mask = state.relabel_mask[indices].to(device)
        unlabeled_mask = state.unlabeled_mask[indices].to(device)
        low_uncertainty_probability = state.low_uncertainty_probability[indices].to(device)
        model_probability = state.model_probability[indices].to(device)
        logits = model(x)
        numerator = torch.zeros((), device=device)
        sample_count = clean_mask.sum() + relabel_mask.sum() + unlabeled_mask.sum()

        if clean_mask.any():
            numerator = numerator + F.cross_entropy(
                logits[clean_mask], observed_y[clean_mask], reduction="sum"
            )
        if relabel_mask.any():
            q = low_uncertainty_probability[relabel_mask].unsqueeze(1)
            original = F.one_hot(observed_y[relabel_mask], NUM_CLASSES).float()
            soft_target = q * model_probability[relabel_mask] + (1.0 - q) * original
            numerator = numerator - (
                soft_target * F.log_softmax(logits[relabel_mask], dim=1)
            ).sum()
        if unlabeled_mask.any():
            weak_x = _augment(
                x[unlabeled_mask],
                args.weak_noise,
                args.weak_scale,
                0.0,
            )
            strong_x = _augment(
                x[unlabeled_mask],
                args.strong_noise,
                args.strong_scale,
                args.strong_mask_fraction,
            )
            weak_probability = model(weak_x).softmax(dim=1).detach()
            strong_probability = model(strong_x).softmax(dim=1)
            numerator = numerator + args.consistency_weight * F.mse_loss(
                strong_probability, weak_probability, reduction="sum"
            )

        loss = numerator / sample_count.clamp_min(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def replay_train(model, optimizer, memory: PuriDivERMemory, args, task_index: int) -> list[dict]:
    rows = []
    for epoch in range(args.replay_epochs):
        epoch_seed = args.seed + 100_000 * task_index + epoch
        if args.method == "puridiver" and epoch >= args.warmup_epochs:
            state = puridiver_split(
                model,
                memory.pool,
                args.device,
                args.infer_batch_size,
                epoch_seed,
            )
            loss = puridiver_replay_epoch(
                model, optimizer, memory.pool, state, args, epoch_seed
            )
            row = {"epoch": epoch + 1, "loss": loss, **state.diagnostics}
        else:
            loss = standard_replay_epoch(model, optimizer, memory.pool, args, epoch_seed)
            row = {"epoch": epoch + 1, "loss": loss, "mode": "warmup_or_er"}
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate(model: nn.Module, pool: EpochPool, args) -> dict:
    loader = DataLoader(
        PoolDataset(pool), batch_size=args.infer_batch_size, shuffle=False, num_workers=0
    )
    predictions, labels = [], []
    model.eval()
    for x, _observed_y, true_y, _indices in loader:
        predictions.append(model(x.to(args.device)).argmax(dim=1).cpu().numpy())
        labels.append(true_y.numpy())
    prediction = np.concatenate(predictions)
    label = np.concatenate(labels)
    return {
        "acc": float(accuracy_score(label, prediction)),
        "mf1": float(
            f1_score(label, prediction, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
        ),
        "epochs": int(label.size),
    }


def summarize_matrix(matrix: np.ndarray) -> dict:
    task_count = matrix.shape[0]
    diagonal = np.diag(matrix)
    final = matrix[-1, :]
    forgetting = []
    for subject_index in range(task_count - 1):
        history = matrix[subject_index:, subject_index]
        forgetting.append(float(np.nanmax(history) - final[subject_index]))
    all_seen_curve = [float(np.nanmean(matrix[t, : t + 1])) for t in range(task_count)]
    return {
        "final_seen_subject_mean": float(np.nanmean(final)),
        "mean_after_learning_subject": float(np.nanmean(diagonal)),
        "average_seen_subject_curve": float(np.mean(all_seen_curve)),
        "mean_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "all_seen_curve": all_seen_curve,
    }


def serializable_args(args) -> dict:
    payload = vars(args).copy()
    payload["device"] = str(args.device)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def save_metrics(output_root: Path, payload: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def run(args) -> dict:
    fix_seed(args.seed)
    subjects = discover_subjects(args.data_root)
    split = reference_subject_split(subjects, args.seed)
    order = split["new_order"]
    if args.max_subjects > 0:
        order = order[: args.max_subjects]

    model = CompactEEGClassifier().to(args.device)
    initial_parameter_norm = float(
        torch.sqrt(sum(parameter.detach().pow(2).sum() for parameter in model.parameters())).cpu()
    )
    memory = PuriDivERMemory(args.memory_size, args.seed)
    test_pools: list[EpochPool] = []
    acc_matrix = np.full((len(order), len(order)), np.nan, dtype=np.float64)
    mf1_matrix = np.full_like(acc_matrix, np.nan)
    task_rows = []

    payload = {
        "config": serializable_args(args),
        "split": split,
        "order": order,
        "initialization": {
            "type": "random_pytorch_initialization",
            "parameter_norm": initial_parameter_norm,
            "pretrained_checkpoint": None,
            "teacher_model": None,
        },
        "tasks": task_rows,
    }
    save_metrics(args.output_root, payload)

    for task_index, subject in enumerate(order):
        train_paths, test_paths = split_subject_paths(args.data_root, subject, args.train_fraction)
        train_pool, train_stats = load_epoch_pool(
            args.data_root,
            subject,
            train_paths,
            args.label_noise_rate,
            args.seed + 10_000 * (task_index + 1),
        )
        test_pool, test_stats = load_epoch_pool(
            args.data_root,
            subject,
            test_paths,
            0.0,
            args.seed,
        )
        before = evaluate(model, test_pool, args)
        optimizer = make_optimizer(model, args.learning_rate)
        online = online_train_subject(
            model, optimizer, train_pool, memory, args, task_index + 1
        )
        replay = replay_train(model, optimizer, memory, args, task_index + 1)
        test_pools.append(test_pool)

        for seen_index, seen_pool in enumerate(test_pools):
            result = evaluate(model, seen_pool, args)
            acc_matrix[task_index, seen_index] = result["acc"]
            mf1_matrix[task_index, seen_index] = result["mf1"]
        after = {
            "acc": float(acc_matrix[task_index, task_index]),
            "mf1": float(mf1_matrix[task_index, task_index]),
            "epochs": len(test_pool),
        }
        old_acc = (
            float(np.nanmean(acc_matrix[task_index, :task_index])) if task_index else None
        )
        old_mf1 = (
            float(np.nanmean(mf1_matrix[task_index, :task_index])) if task_index else None
        )
        row = {
            "task": task_index + 1,
            "subject": subject,
            "train": train_stats,
            "test": test_stats,
            "mini_batches_expected": int(math.ceil(len(train_pool) / args.online_batch_size)),
            "before": before,
            "after": after,
            "plasticity": {
                "acc_gain": after["acc"] - before["acc"],
                "mf1_gain": after["mf1"] - before["mf1"],
            },
            "old_subject_mean_after": {"acc": old_acc, "mf1": old_mf1},
            "all_seen_mean_after": {
                "acc": float(np.nanmean(acc_matrix[task_index, : task_index + 1])),
                "mf1": float(np.nanmean(mf1_matrix[task_index, : task_index + 1])),
            },
            "online": online,
            "replay": replay,
            "memory": {
                "size": len(memory),
                "purity": diagnostic_label_purity(memory.pool),
                "class_counts": torch.bincount(
                    memory.pool.observed_y, minlength=NUM_CLASSES
                ).tolist(),
            },
        }
        task_rows.append(row)
        payload["acc_matrix"] = acc_matrix.tolist()
        payload["mf1_matrix"] = mf1_matrix.tolist()
        payload["summary"] = {
            "acc": summarize_matrix(acc_matrix[: task_index + 1, : task_index + 1]),
            "mf1": summarize_matrix(mf1_matrix[: task_index + 1, : task_index + 1]),
            "mean_new_subject_acc_gain": float(
                np.mean([task["plasticity"]["acc_gain"] for task in task_rows])
            ),
            "mean_new_subject_mf1_gain": float(
                np.mean([task["plasticity"]["mf1_gain"] for task in task_rows])
            ),
        }
        save_metrics(args.output_root, payload)
        print(
            f"[{args.method}] task={task_index + 1}/{len(order)} subject={subject} "
            f"batches={online['mini_batches']} new_acc={before['acc']:.4f}->{after['acc']:.4f} "
            f"old_acc={old_acc if old_acc is not None else float('nan'):.4f} "
            f"seen_acc={row['all_seen_mean_after']['acc']:.4f} memory={len(memory)} "
            f"purity={row['memory']['purity']:.4f}",
            flush=True,
        )

    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "rttdp_brainuicl_runs" / "pure_puridiver_eeg",
    )
    parser.add_argument(
        "--method",
        choices=["puridiver", "puri_memory", "er"],
        default="puridiver",
        help="puri_memory uses PuriDivER sampling with ordinary CE replay.",
    )
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--label-noise-rate", type=float, default=0.0)
    parser.add_argument("--memory-size", type=int, default=1000, help="Number of 30-second epochs.")
    parser.add_argument("--online-batch-size", type=int, default=128)
    parser.add_argument("--replay-batch-size", type=int, default=128)
    parser.add_argument("--infer-batch-size", type=int, default=256)
    parser.add_argument("--replay-epochs", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-diversity-coefficient", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--weak-noise", type=float, default=0.01)
    parser.add_argument("--weak-scale", type=float, default=0.02)
    parser.add_argument("--strong-noise", type=float, default=0.05)
    parser.add_argument("--strong-scale", type=float, default=0.10)
    parser.add_argument("--strong-mask-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be in (0, 1)")
    if not 0.0 <= args.label_noise_rate < 1.0:
        parser.error("--label-noise-rate must be in [0, 1)")
    args.device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    return args


if __name__ == "__main__":
    run(parse_args())
