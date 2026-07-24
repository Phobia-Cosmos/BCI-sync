"""Signal-only EEG datasets for target continual-learning updates."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class UnlabeledSequenceDataset(Dataset):
    """Load uploaded signals without accepting or opening annotation paths."""

    def __init__(self, data_paths: Sequence[Path], sequence_length: int):
        self.data_paths = [Path(path) for path in data_paths]
        self.sequence_length = int(sequence_length)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int):
        values = np.load(self.data_paths[index], allow_pickle=False)
        if values.ndim != 3 or values.shape[0] != self.sequence_length:
            raise ValueError(
                f"Expected [{self.sequence_length},channels,time] EEG sequence: "
                f"{self.data_paths[index]}"
            )
        values = torch.from_numpy(values.astype(np.float32, copy=False))
        dummy = torch.zeros(self.sequence_length, dtype=torch.long)
        return values[:, :2, :], values[:, 2:, :], dummy


__all__ = ["UnlabeledSequenceDataset"]
