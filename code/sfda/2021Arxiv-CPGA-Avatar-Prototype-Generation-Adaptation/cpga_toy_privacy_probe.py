#!/usr/bin/env python3
"""Minimal CPGA-style privacy probe using only NumPy.

This is not an official CPGA implementation. It demonstrates the key privacy
intuition behind avatar prototype generation: a released source model can carry
enough information to synthesize source-like class prototypes.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def make_source_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [-3.0, -1.0],
            [0.4, 3.0],
            [3.2, -1.2],
        ],
        dtype=np.float64,
    )
    samples_per_class = 240
    xs = []
    ys = []
    for label, center in enumerate(centers):
        xs.append(center + rng.normal(scale=0.55, size=(samples_per_class, 2)))
        ys.append(np.full(samples_per_class, label, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ys), centers


def train_source_model(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    lr: float = 0.08,
    steps: int = 2500,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a small softmax classifier."""

    rng = np.random.default_rng(11)
    w = rng.normal(scale=0.1, size=(x.shape[1], num_classes))
    b = np.zeros(num_classes, dtype=np.float64)
    one_hot = np.eye(num_classes)[y]
    n = x.shape[0]

    for _ in range(steps):
        probs = softmax(x @ w + b)
        grad_logits = (probs - one_hot) / n
        grad_w = x.T @ grad_logits
        grad_b = grad_logits.sum(axis=0)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def generate_avatar_prototypes(
    w: np.ndarray,
    b: np.ndarray,
    num_classes: int,
    lr: float = 0.08,
    steps: int = 1500,
    l2: float = 0.015,
) -> np.ndarray:
    """Generate one avatar vector per class using only the frozen model.

    Objective per class:
      CE(source_model(z), target_class) + l2 * ||z||^2

    This is a simplified analogue of source avatar prototype mining.
    """

    rng = np.random.default_rng(23)
    avatars = rng.normal(scale=0.2, size=(num_classes, w.shape[0]))

    for _ in range(steps):
        logits = avatars @ w + b
        probs = softmax(logits)
        target = np.eye(num_classes)
        grad_logits = probs - target
        grad_avatars = grad_logits @ w.T + 2.0 * l2 * avatars
        avatars -= lr * grad_avatars
    return avatars


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def main() -> None:
    x, y, hidden_centers = make_source_data()
    num_classes = len(hidden_centers)
    w, b = train_source_model(x, y, num_classes)

    # Source data is discarded here. The attacker/probe only uses w and b.
    avatars = generate_avatar_prototypes(w, b, num_classes)
    avatar_probs = softmax(avatars @ w + b)
    distances = pairwise_dist(avatars, hidden_centers)

    print("Hidden source class centers:")
    print(np.round(hidden_centers, 3))
    print("\nGenerated avatar prototypes using only the frozen source model:")
    print(np.round(avatars, 3))
    print("\nSource-model confidence for generated avatars:")
    print(np.round(avatar_probs, 4))
    print("\nDistance from each avatar to each hidden source center:")
    print(np.round(distances, 3))
    print("\nNearest hidden center for each avatar:", distances.argmin(axis=1).tolist())
    print("Mean nearest-center distance:", round(distances.min(axis=1).mean(), 4))

    recovered = np.array_equal(distances.argmin(axis=1), np.arange(num_classes))
    print("\nPrivacy interpretation:")
    if recovered:
        print(
            "Recovered class-level source prototypes. This supports the claim "
            "that a source model can leak source distribution information even "
            "when raw source data is unavailable."
        )
    else:
        print(
            "The probe did not recover all centers in this run. Try adjusting "
            "regularization/steps or use a richer source model."
        )


if __name__ == "__main__":
    main()

