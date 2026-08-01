"""Auditable state and direction utilities for the PERSIST-EEG probe.

This module is deliberately victim-independent.  It consumes only probability
responses and proxy-side input directions, so it cannot access victim
parameters, optimizer state, or replay memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0)
    return float((-values * np.log(values)).sum(axis=-1).mean())


def _margin(probabilities: np.ndarray) -> float:
    values = np.sort(np.asarray(probabilities, dtype=np.float64), axis=-1)
    return float((values[..., -1] - values[..., -2]).mean())


def _mean_kl(left: np.ndarray, right: np.ndarray) -> float:
    a = np.clip(np.asarray(left, dtype=np.float64), 1e-8, 1.0)
    b = np.clip(np.asarray(right, dtype=np.float64), 1e-8, 1.0)
    return float((a * (np.log(a) - np.log(b))).sum(axis=-1).mean())


@dataclass
class ProbabilityStateFilter:
    """Short/mid/long probability state from owned upload responses."""

    short_alpha: float = 0.5
    mid_alpha: float = 0.2
    long_alpha: float = 0.05
    short: np.ndarray | None = None
    middle: np.ndarray | None = None
    long: np.ndarray | None = None
    observations: int = 0
    last: dict[str, float] = field(default_factory=dict)

    def update(self, probabilities: np.ndarray) -> dict[str, float]:
        values = np.asarray(probabilities, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(f"Expected [sequence, epoch, class], got {values.shape}")
        mean_probability = values.mean(axis=0, keepdims=True)
        if self.short is None:
            self.short = mean_probability.copy()
            self.middle = mean_probability.copy()
            self.long = mean_probability.copy()
            kl_short = kl_long = 0.0
        else:
            kl_short = _mean_kl(mean_probability, self.short)
            kl_long = _mean_kl(mean_probability, self.long)
            self.short = self.short_alpha * mean_probability + (1 - self.short_alpha) * self.short
            self.middle = self.mid_alpha * mean_probability + (1 - self.mid_alpha) * self.middle
            self.long = self.long_alpha * mean_probability + (1 - self.long_alpha) * self.long
        self.observations += 1
        self.last = {
            "mean_entropy": _entropy(values),
            "mean_margin": _margin(values),
            "kl_to_short": float(kl_short),
            "kl_to_long": float(kl_long),
            "observations": float(self.observations),
        }
        return dict(self.last)

    def state(self) -> dict[str, Any]:
        return {"observations": self.observations, **self.last}


@dataclass
class DirectionBank:
    """Bounded EMA bank of proxy-side input directions."""

    capacity: int = 4
    decay: float = 0.8
    directions: list[np.ndarray] = field(default_factory=list)

    @staticmethod
    def _canonical(direction: np.ndarray) -> np.ndarray:
        value = np.asarray(direction, dtype=np.float32)
        if value.ndim < 2:
            return value.copy()
        # Sequence cardinality changes across subjects; keep the shared
        # epoch/channel/time direction and average only the sequence axis.
        return value.mean(axis=0).astype(np.float32, copy=False)

    def update(self, direction: np.ndarray) -> dict[str, float | int | None]:
        current = self._canonical(direction)
        if not np.isfinite(current).all():
            raise ValueError("Direction contains non-finite values")
        if not self.directions:
            self.directions.append(current.copy())
        else:
            blended = self.decay * self.directions[-1] + (1.0 - self.decay) * current
            self.directions.append(blended.astype(np.float32, copy=False))
            if len(self.directions) > self.capacity:
                self.directions.pop(0)
        norm = float(np.linalg.norm(self.directions[-1].reshape(-1)))
        return {"bank_size": len(self.directions), "bank_norm": norm}

    def mean(self) -> np.ndarray | None:
        if not self.directions:
            return None
        return np.mean(np.stack(self.directions, axis=0), axis=0).astype(np.float32)

    def cosine(self, direction: np.ndarray) -> float | None:
        reference = self.mean()
        if reference is None:
            return None
        left = self._canonical(direction).astype(np.float64).reshape(-1)
        right = reference.astype(np.float64).reshape(-1)
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        return float(np.dot(left, right) / max(denominator, 1e-12))

    def state(self) -> dict[str, Any]:
        reference = self.mean()
        return {
            "capacity": self.capacity,
            "decay": self.decay,
            "bank_size": len(self.directions),
            "mean_norm": float(np.linalg.norm(reference.reshape(-1))) if reference is not None else 0.0,
        }


def proxy_information_score(probabilities: np.ndarray, direction_cosine: float | None) -> dict[str, float]:
    """Return pre-upload diagnostics; no victim state is used."""

    values = np.asarray(probabilities, dtype=np.float32)
    uncertainty = _entropy(values)
    margin = _margin(values)
    return {
        "information_entropy": uncertainty,
        "information_margin": margin,
        "information_score": uncertainty + (1.0 - margin),
        "direction_bank_cosine": float(direction_cosine) if direction_cosine is not None else 0.0,
    }
