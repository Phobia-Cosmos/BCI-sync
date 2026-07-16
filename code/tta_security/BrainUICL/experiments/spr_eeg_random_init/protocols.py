from __future__ import annotations

from pathlib import Path

import numpy as np


PathPair = tuple[list[Path], list[Path]]


def split_sequence_paths(paths: PathPair, holdout_ratio: float, seed: int) -> tuple[PathPair, PathPair]:
    """Deterministically split non-overlapping sequences for adaptation and evaluation."""

    data_paths, label_paths = paths
    if len(data_paths) != len(label_paths):
        raise ValueError("data and label path counts differ")
    if len(data_paths) < 2:
        raise ValueError("a subject needs at least two sequences for a holdout split")
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data_paths))
    holdout_count = min(max(1, int(round(len(indices) * holdout_ratio))), len(indices) - 1)
    test_indices = sorted(int(index) for index in indices[:holdout_count])
    train_indices = sorted(int(index) for index in indices[holdout_count:])
    train = ([data_paths[index] for index in train_indices], [label_paths[index] for index in train_indices])
    test = ([data_paths[index] for index in test_indices], [label_paths[index] for index in test_indices])
    return train, test


def shuffled_subject_order(subjects: list[int], seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    order = np.asarray(subjects, dtype=np.int64)
    rng.shuffle(order)
    return [int(subject) for subject in order]

