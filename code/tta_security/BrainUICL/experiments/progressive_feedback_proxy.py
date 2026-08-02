"""Online score-feedback proxy stream for the aligned EEG CL runners.

The controller never receives victim parameters. It sees only public output
probabilities for data uploaded at the current task, then updates an independent
surrogate initialized from the same public pretrain checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .regularization_cl_attacks import (
        proxy_dual_harm_batch,
        pseudo_update_gradients,
        supervised_update_gradients,
    )
    from .rttdp_brainuicl_full import (
        CPCProbe,
        clone_blocks,
        flat_logits,
        forward_blocks,
        load_pretrained,
        set_train,
        subject_paths,
    )
    from .unlabeled_eeg import UnlabeledSequenceDataset
except ImportError:  # Direct runner execution from experiments/.
    from regularization_cl_attacks import (
        proxy_dual_harm_batch,
        pseudo_update_gradients,
        supervised_update_gradients,
    )
    from rttdp_brainuicl_full import (
        CPCProbe,
        clone_blocks,
        flat_logits,
        forward_blocks,
        load_pretrained,
        set_train,
        subject_paths,
    )
    from unlabeled_eeg import UnlabeledSequenceDataset

from model.regularization_cl import freeze_batch_norm_running_stats
try:
    from .persist_eeg import DirectionBank, ProbabilityStateFilter, proxy_information_score
except ImportError:
    from persist_eeg import DirectionBank, ProbabilityStateFilter, proxy_information_score


PROGRESSIVE_PROXY_MODES = ("none", "static", "feedback", "population_feedback")


def add_progressive_proxy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--progressive-persist", action="store_true", help="Enable auditable PERSIST-EEG state diagnostics.")
    parser.add_argument("--progressive-direction-bank-capacity", type=int, default=4)
    parser.add_argument("--progressive-direction-bank-decay", type=float, default=0.8)
    parser.add_argument(
        "--progressive-proxy-mode",
        choices=PROGRESSIVE_PROXY_MODES,
        default="none",
    )
    parser.add_argument(
        "--progressive-proxy-tasks",
        default="odd",
        help="odd, even, or comma-separated 1-based task indices.",
    )
    parser.add_argument(
        "--progressive-clean-feedback-tasks",
        default="even",
        help="Clean tasks whose owned signals and returned scores are visible.",
    )
    parser.add_argument("--progressive-base-subject", type=int, default=18)
    parser.add_argument(
        "--progressive-population-refresh-mix",
        type=float,
        default=0.20,
        help="Blend fraction of the current class/subject-balanced buffer pool at each Proxy task.",
    )
    parser.add_argument("--progressive-population-candidates-per-class", type=int, default=4)
    parser.add_argument(
        "--progressive-population-cross-class-mix",
        type=float,
        default=0.0,
        help="Fraction of invariant statistics fitted from the most confusable class.",
    )
    parser.add_argument("--progressive-preserve-eeg-invariants", action="store_true")
    parser.add_argument(
        "--progressive-invariant-drift-tolerance",
        type=float,
        default=0.02,
        help="Maximum relative drift of the physiological EEG descriptor.",
    )
    parser.add_argument("--progressive-proxy-lr", type=float, default=1e-6)
    parser.add_argument("--progressive-feedback-steps", type=int, default=4)
    parser.add_argument("--progressive-feedback-batch", type=int, default=4)
    parser.add_argument("--progressive-guide-epochs", type=int, default=2)
    parser.add_argument("--progressive-generation-steps", type=int, default=3)
    parser.add_argument(
        "--progressive-generation-attempts",
        type=int,
        default=1,
        help="Regenerate a rejected candidate without relaxing the source gate.",
    )
    parser.add_argument("--progressive-generation-batch", type=int, default=4)
    parser.add_argument("--progressive-reference-batch", type=int, default=4)
    parser.add_argument("--progressive-step-relative-l2", type=float, default=0.01)
    parser.add_argument("--progressive-step-linf-std", type=float, default=0.025)
    parser.add_argument(
        "--progressive-cumulative-relative-l2", type=float, default=0.20
    )
    parser.add_argument(
        "--progressive-cumulative-linf-std", type=float, default=0.50
    )
    parser.add_argument("--progressive-history-weight", type=float, default=1.0)
    parser.add_argument("--progressive-history-decay", type=float, default=0.8)
    parser.add_argument("--progressive-input-cone-residual", type=float, default=0.5)
    parser.add_argument("--progressive-source-weight", type=float, default=0.5)
    parser.add_argument("--progressive-feedback-weight", type=float, default=1.0)
    parser.add_argument("--progressive-feedback-decay", type=float, default=0.95)
    parser.add_argument("--progressive-feedback-capacity", type=int, default=2500)
    parser.add_argument(
        "--progressive-upload-full-pool",
        action="store_true",
        help="Upload all fixed-subject sequences at each proxy task.",
    )
    parser.add_argument(
        "--progressive-match-task-sequence-count",
        action="store_true",
        help=(
            "Upload exactly as many Proxy sequences as the clean subject at "
            "the corresponding task."
        ),
    )
    parser.add_argument(
        "--progressive-require-all-sequences-modified",
        action="store_true",
        help=(
            "Fail a feedback Proxy task unless every uploaded sequence differs "
            "from its initial clean sequence."
        ),
    )
    parser.add_argument("--progressive-active-fraction", type=float, default=1.0)
    parser.add_argument("--progressive-passive-step-scale", type=float, default=1.0)
    parser.add_argument(
        "--progressive-passive-cumulative-relative-l2", type=float, default=-1.0
    )
    parser.add_argument(
        "--progressive-passive-cumulative-linf-std", type=float, default=-1.0
    )
    parser.add_argument("--progressive-fill-step-budget", action="store_true")
    parser.add_argument("--progressive-history-refresh-count", type=int, default=0)
    parser.add_argument("--progressive-target-weight", type=float, default=-1.0)
    parser.add_argument("--progressive-conflict-weight", type=float, default=-1.0)
    parser.add_argument("--progressive-gradient-norm-weight", type=float, default=-1.0)
    parser.add_argument("--progressive-virtual-old-weight", type=float, default=-1.0)
    parser.add_argument("--progressive-virtual-new-weight", type=float, default=-1.0)
    parser.add_argument("--progressive-confidence-weight", type=float, default=-1.0)
    parser.add_argument("--progressive-l2-weight", type=float, default=-1.0)
    parser.add_argument(
        "--progressive-max-source-gradient-cosine", type=float, default=1.0
    )
    parser.add_argument("--progressive-source-gate-samples", type=int, default=0)
    parser.add_argument("--progressive-require-source-conflict", action="store_true")
    parser.add_argument("--progressive-survival-trajectories", type=int, default=0)
    parser.add_argument("--progressive-survival-steps", type=int, default=0)
    parser.add_argument("--progressive-survival-batch", type=int, default=4)
    parser.add_argument("--progressive-survival-weight", type=float, default=0.0)
    parser.add_argument("--progressive-survival-temperature", type=float, default=0.25)


def validate_progressive_proxy_args(args, total_tasks: int) -> None:
    if args.progressive_proxy_mode == "none":
        return
    if getattr(args, "progressive_direction_bank_capacity", 4) < 1:
        raise ValueError("Progressive direction bank capacity must be positive")
    positive_names = (
        "progressive_proxy_lr",
        "progressive_feedback_batch",
        "progressive_generation_steps",
        "progressive_generation_batch",
        "progressive_reference_batch",
        "progressive_step_relative_l2",
        "progressive_step_linf_std",
        "progressive_cumulative_relative_l2",
        "progressive_cumulative_linf_std",
        "progressive_feedback_capacity",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.progressive_feedback_steps < 0 or args.progressive_guide_epochs < 0:
        raise ValueError("Progressive feedback/guide epochs cannot be negative")
    if getattr(args, "progressive_generation_attempts", 1) < 1:
        raise ValueError("Progressive generation attempts must be positive")
    if not 0.0 <= args.progressive_history_decay <= 1.0:
        raise ValueError("Progressive history decay must be in [0, 1]")
    if not 0.0 < args.progressive_feedback_decay <= 1.0:
        raise ValueError("Progressive feedback decay must be in (0, 1]")
    if not 0.0 <= args.progressive_input_cone_residual <= 1.0:
        raise ValueError("Progressive input cone residual must be in [0, 1]")
    if not 0.0 < getattr(args, "progressive_active_fraction", 1.0) <= 1.0:
        raise ValueError("Progressive active fraction must be in (0, 1]")
    if not 0.0 <= getattr(args, "progressive_passive_step_scale", 1.0) <= 1.0:
        raise ValueError("Progressive passive step scale must be in [0, 1]")
    if getattr(args, "progressive_history_refresh_count", 0) < 0:
        raise ValueError("Progressive history refresh count cannot be negative")
    if not 0.0 <= getattr(args, "progressive_population_refresh_mix", 0.20) <= 1.0:
        raise ValueError("Population refresh mix must be in [0, 1]")
    if getattr(args, "progressive_population_candidates_per_class", 4) < 2:
        raise ValueError("Population candidates per class must be at least two")
    if not 0.0 <= getattr(args, "progressive_population_cross_class_mix", 0.0) < 0.5:
        raise ValueError("Population cross-class mix must be in [0, 0.5)")
    if not 0.0 < getattr(args, "progressive_invariant_drift_tolerance", 0.02) < 1.0:
        raise ValueError("Population invariant drift tolerance must be in (0, 1)")
    survival_values = (
        getattr(args, "progressive_survival_trajectories", 0),
        getattr(args, "progressive_survival_steps", 0),
        getattr(args, "progressive_survival_batch", 4),
    )
    if any(value < 0 for value in survival_values):
        raise ValueError("Progressive survival counts cannot be negative")
    survival_weight = getattr(args, "progressive_survival_weight", 0.0)
    survival_enabled = survival_weight > 0.0
    if survival_enabled and (
        survival_values[0] <= 0
        or survival_values[1] <= 0
        or survival_values[2] <= 0
    ):
        raise ValueError(
            "Positive progressive survival weight requires positive "
            "trajectories, steps, and batch"
        )
    if not survival_enabled and (
        survival_values[0] > 0
        or survival_values[1] > 0
    ):
        raise ValueError(
            "Progressive survival trajectories/steps require a positive weight"
        )
    if getattr(args, "progressive_survival_temperature", 0.25) <= 0.0:
        raise ValueError("Progressive survival temperature must be positive")
    if getattr(args, "progressive_source_gate_samples", 0) < 0:
        raise ValueError("Progressive source gate samples cannot be negative")
    if not -1.0 <= getattr(args, "progressive_max_source_gradient_cosine", 1.0) <= 1.0:
        raise ValueError("Progressive source-gradient cosine must be in [-1, 1]")
    proxy_tasks = resolve_task_spec(args.progressive_proxy_tasks, total_tasks)
    clean_tasks = resolve_task_spec(
        args.progressive_clean_feedback_tasks,
        total_tasks,
    )
    overlap = proxy_tasks & clean_tasks
    if overlap:
        raise ValueError(f"Progressive proxy/clean-feedback tasks overlap: {sorted(overlap)}")
    if args.progressive_proxy_mode in ("feedback", "population_feedback"):
        full_pool = getattr(args, "progressive_upload_full_pool", False)
        match_task = getattr(
            args,
            "progressive_match_task_sequence_count",
            False,
        )
        if full_pool == match_task:
            raise ValueError(
                "Feedback Proxy requires exactly one upload-cardinality mode: "
                "--progressive-upload-full-pool or "
                "--progressive-match-task-sequence-count"
            )
        if getattr(args, "progressive_active_fraction", 1.0) != 1.0:
            raise ValueError(
                "Feedback Proxy requires --progressive-active-fraction 1 so every "
                "uploaded sequence is actively modified"
            )


def resolve_task_spec(value: str, total_tasks: int) -> set[int]:
    text = value.strip().lower()
    if text == "odd":
        return set(range(1, total_tasks + 1, 2))
    if text == "even":
        return set(range(2, total_tasks + 1, 2))
    if not text:
        return set()
    tasks = {int(item.strip()) for item in text.split(",") if item.strip()}
    invalid = sorted(task for task in tasks if not 1 <= task <= total_tasks)
    if invalid:
        raise ValueError(f"Progressive task indices outside 1..{total_tasks}: {invalid}")
    return tasks


class ArrayUnlabeledDataset(Dataset):
    def __init__(self, arrays: np.ndarray):
        self.arrays = arrays

    def __len__(self) -> int:
        return int(self.arrays.shape[0])

    def __getitem__(self, index: int):
        tensor = torch.from_numpy(np.asarray(self.arrays[index], dtype=np.float32))
        dummy = torch.zeros(tensor.shape[0], dtype=torch.long)
        split = 1 if tensor.shape[1] == 32 else 2
        return tensor[:, :split, :], tensor[:, split:, :], dummy


@torch.no_grad()
def public_probabilities(blocks, data_paths: Sequence[Path], args) -> np.ndarray:
    loader = DataLoader(
        UnlabeledSequenceDataset(list(data_paths), args.model_param.SeqLength),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    modes = [block.training for block in blocks]
    set_train(blocks, False)
    rows: list[np.ndarray] = []
    for eog, eeg, _dummy in loader:
        logits = forward_blocks(
            blocks,
            eog.to(args.device),
            eeg.to(args.device),
            args,
        )
        rows.append(logits.softmax(dim=1).permute(0, 2, 1).cpu().numpy())
    for block, mode in zip(blocks, modes):
        block.train(mode)
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def _load_arrays(items: Sequence[Path | np.ndarray]) -> np.ndarray:
    arrays = [
        np.load(item, allow_pickle=False).astype(np.float32, copy=False)
        if isinstance(item, Path)
        else np.asarray(item, dtype=np.float32)
        for item in items
    ]
    return np.stack(arrays)


def _eeg_invariant_descriptor(array: np.ndarray) -> np.ndarray:
    """Compact class/subject descriptor without retaining waveform phase."""
    signal = np.asarray(array, dtype=np.float32)
    downsampled = signal[..., ::8]
    centered = downsampled - downsampled.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=-1)
    power = np.abs(spectrum).astype(np.float64) ** 2
    frequency_count = power.shape[-1] - 1
    frequency_bins = np.array_split(
        power[..., 1:], min(6, max(frequency_count, 1)), axis=-1
    )
    bandpower = np.asarray(
        [float(local.mean()) for local in frequency_bins],
        dtype=np.float64,
    )
    if len(bandpower) < 6:
        bandpower = np.pad(bandpower, (0, 6 - len(bandpower)))
    bandpower = np.log1p(bandpower)
    bandpower -= bandpower.mean()
    bandpower /= max(float(np.linalg.norm(bandpower)), 1e-12)

    channels = centered.transpose(1, 0, 2).reshape(centered.shape[1], -1)
    covariance = channels @ channels.T / max(channels.shape[1] - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
    eigenvalues = eigenvalues[: min(6, len(eigenvalues))]
    eigenvalues /= max(float(eigenvalues.sum()), 1e-12)
    if len(eigenvalues) < 6:
        eigenvalues = np.pad(eigenvalues, (0, 6 - len(eigenvalues)))

    autocorrelation: list[float] = []
    variance = float(np.mean(centered * centered)) + 1e-12
    for lag in (1, 4, 16):
        if centered.shape[-1] <= lag:
            autocorrelation.append(0.0)
        else:
            autocorrelation.append(
                float(np.mean(centered[..., :-lag] * centered[..., lag:]) / variance)
            )
    return np.concatenate((bandpower, eigenvalues, np.asarray(autocorrelation)))


def physiological_eeg_descriptor(array: np.ndarray, dataset: str) -> np.ndarray:
    """Dataset-calibrated EEG statistics shared by ISRUC and FACED."""
    signal = np.asarray(array, dtype=np.float64)
    signal = signal if dataset == "FACED" else signal[:, 2:, :]
    sampling_rate = 250.0 if dataset == "FACED" else 100.0
    centered = signal - signal.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=-1)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(centered.shape[-1], d=1.0 / sampling_rate)
    bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
    bandpower = np.asarray([
        float(power[..., (frequencies >= low) & (frequencies < high)].mean())
        for low, high in bands
    ])
    bandpower /= max(float(bandpower.sum()), 1e-12)
    channels = centered.transpose(1, 0, 2).reshape(centered.shape[1], -1)
    covariance = channels @ channels.T / max(channels.shape[1] - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1][:6]
    eigenvalues /= max(float(eigenvalues.sum()), 1e-12)
    eigenvalues = np.pad(eigenvalues, (0, 6 - len(eigenvalues)))
    variance = float(np.mean(centered * centered)) + 1e-12
    autocorrelation = []
    for seconds in (0.01, 0.04, 0.16):
        lag = max(1, int(round(seconds * sampling_rate)))
        autocorrelation.append(
            float(np.mean(centered[..., :-lag] * centered[..., lag:]) / variance)
        )
    return np.concatenate((bandpower, eigenvalues, np.asarray(autocorrelation)))


def invariant_drift(candidate: np.ndarray, reference: np.ndarray, dataset: str) -> float:
    candidate_descriptor = physiological_eeg_descriptor(candidate, dataset)
    reference_descriptor = physiological_eeg_descriptor(reference, dataset)
    return float(
        np.linalg.norm(candidate_descriptor - reference_descriptor)
        / max(np.linalg.norm(reference_descriptor), 1e-12)
    )


def _limit_invariant_drift(
    candidate: np.ndarray,
    reference: np.ndarray,
    dataset: str,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    drift = invariant_drift(candidate, reference, dataset)
    if drift <= tolerance:
        return candidate.astype(np.float32, copy=False), drift
    low, high = 0.0, 1.0
    best = np.asarray(reference, dtype=np.float32)
    best_drift = 0.0
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    for _iteration in range(14):
        scale = (low + high) / 2.0
        trial = (np.asarray(reference, dtype=np.float64) + scale * delta).astype(np.float32)
        trial_drift = invariant_drift(trial, reference, dataset)
        if trial_drift <= tolerance:
            low = scale
            best, best_drift = trial, trial_drift
        else:
            high = scale
    return best, best_drift


def _limit_batch_invariant_drift(
    candidates: np.ndarray,
    references: np.ndarray,
    dataset: str,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, drifts = [], []
    for candidate, reference in zip(candidates, references):
        row, drift = _limit_invariant_drift(candidate, reference, dataset, tolerance)
        rows.append(row)
        drifts.append(drift)
    return np.stack(rows), np.asarray(drifts, dtype=np.float64)


def _match_channel_covariance(
    signal: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    alpha: float,
) -> np.ndarray:
    channels = signal.shape[1]
    sample_matrix = signal.transpose(1, 0, 2).reshape(channels, -1).astype(np.float64)
    first_matrix = first.transpose(1, 0, 2).reshape(channels, -1).astype(np.float64)
    second_matrix = second.transpose(1, 0, 2).reshape(channels, -1).astype(np.float64)
    sample_mean = sample_matrix.mean(axis=1, keepdims=True)
    first_mean = first_matrix.mean(axis=1, keepdims=True)
    second_mean = second_matrix.mean(axis=1, keepdims=True)
    sample_centered = sample_matrix - sample_mean
    first_centered = first_matrix - first_mean
    second_centered = second_matrix - second_mean
    denominator = max(sample_matrix.shape[1] - 1, 1)
    sample_covariance = sample_centered @ sample_centered.T / denominator
    target_covariance = (
        alpha * (first_centered @ first_centered.T / denominator)
        + (1.0 - alpha) * (second_centered @ second_centered.T / denominator)
    )
    sample_values, sample_vectors = np.linalg.eigh(sample_covariance)
    target_values, target_vectors = np.linalg.eigh(target_covariance)
    whitening = sample_vectors @ np.diag(1.0 / np.sqrt(np.maximum(sample_values, 1e-8))) @ sample_vectors.T
    coloring = target_vectors @ np.diag(np.sqrt(np.maximum(target_values, 1e-8))) @ target_vectors.T
    target_mean = alpha * first_mean + (1.0 - alpha) * second_mean
    matched = coloring @ whitening @ sample_centered + target_mean
    return matched.reshape(channels, signal.shape[0], signal.shape[2]).transpose(1, 0, 2).astype(np.float32)


def _fit_eeg_invariants(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Fit spectral magnitude and cross-channel covariance from two EEG sequences."""
    outputs: list[np.ndarray] = []
    split = 1 if first.shape[1] == 32 else 2
    for channel_slice in (slice(0, split), slice(split, None)):
        left = np.asarray(first[:, channel_slice, :], dtype=np.float32)
        right = np.asarray(second[:, channel_slice, :], dtype=np.float32)
        left_spectrum = np.fft.rfft(left, axis=-1)
        right_spectrum = np.fft.rfft(right, axis=-1)
        log_magnitude = (
            alpha * np.log(np.abs(left_spectrum) + 1e-8)
            + (1.0 - alpha) * np.log(np.abs(right_spectrum) + 1e-8)
        )
        phase = np.angle(left_spectrum)
        fitted = np.fft.irfft(
            np.exp(log_magnitude + 1j * phase),
            n=left.shape[-1],
            axis=-1,
        ).astype(np.float32)
        outputs.append(_match_channel_covariance(fitted, left, right, alpha))
    result = np.concatenate(outputs, axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError("Population EEG invariant fit produced non-finite values")
    return result.astype(np.float32, copy=False)


def _split_modalities(arrays: np.ndarray, device: torch.device):
    tensor = torch.from_numpy(np.asarray(arrays, dtype=np.float32)).to(device)
    split = 1 if tensor.shape[2] == 32 else 2
    return tensor[:, :, :split, :], tensor[:, :, split:, :]


def _relative_l2(delta: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(base.reshape(base.shape[0], -1), axis=1),
        1e-12,
    )


def _project_numpy(
    candidate: np.ndarray,
    base: np.ndarray,
    *,
    max_relative_l2: float,
    max_linf_over_std: float,
) -> np.ndarray:
    delta = candidate.astype(np.float64) - base.astype(np.float64)
    scales = np.ones(delta.shape[0], dtype=np.float64)
    for channel_slice in (slice(None), slice(0, 2), slice(2, None)):
        local_base = base[:, :, channel_slice, :].astype(np.float64)
        local_delta = delta[:, :, channel_slice, :]
        delta_norm = np.linalg.norm(local_delta.reshape(delta.shape[0], -1), axis=1)
        base_norm = np.linalg.norm(local_base.reshape(delta.shape[0], -1), axis=1)
        scales = np.minimum(
            scales,
            np.where(
                delta_norm > 0,
                max_relative_l2 * np.maximum(base_norm, 1e-12) / delta_norm,
                1.0,
            ),
        )
        delta_max = np.max(np.abs(local_delta), axis=(1, 2, 3))
        base_std = np.std(local_base, axis=(1, 2, 3))
        scales = np.minimum(
            scales,
            np.where(
                delta_max > 0,
                max_linf_over_std * np.maximum(base_std, 1e-12) / delta_max,
                1.0,
            ),
        )
    scales = np.minimum(scales, 1.0)
    projected = base.astype(np.float64) + delta * scales[:, None, None, None]
    return projected.astype(np.float32)


def _constrain_step_to_direction(
    step: np.ndarray,
    direction: np.ndarray,
    residual_ratio: float,
) -> tuple[np.ndarray, float]:
    step_flat = step.reshape(step.shape[0], -1).astype(np.float64)
    direction_flat = direction.reshape(direction.shape[0], -1).astype(np.float64)
    output = np.empty_like(step_flat)
    cosines: list[float] = []
    for index, (local_step, local_direction) in enumerate(
        zip(step_flat, direction_flat)
    ):
        direction_norm = np.linalg.norm(local_direction)
        step_norm = np.linalg.norm(local_step)
        if direction_norm <= 1e-12 or step_norm <= 1e-12:
            output[index] = local_step
            cosines.append(0.0)
            continue
        unit = local_direction / direction_norm
        parallel_size = max(float(np.dot(local_step, unit)), step_norm * 0.25)
        parallel = parallel_size * unit
        residual = local_step - float(np.dot(local_step, unit)) * unit
        residual_norm = np.linalg.norm(residual)
        max_residual = residual_ratio * max(parallel_size, 1e-12)
        if residual_norm > max_residual:
            residual *= max_residual / residual_norm
        constrained = parallel + residual
        output[index] = constrained
        cosines.append(
            float(
                np.dot(constrained, local_direction)
                / max(np.linalg.norm(constrained) * direction_norm, 1e-12)
            )
        )
    return output.reshape(step.shape).astype(np.float32), float(np.mean(cosines))


def _fill_step_relative_l2(
    step: np.ndarray,
    base: np.ndarray,
    target_relative_l2: float,
) -> np.ndarray:
    """Rescale every nonzero sequence step to a requested relative-L2 length."""
    step64 = step.astype(np.float64)
    base64 = base.astype(np.float64)
    step_norm = np.linalg.norm(step64.reshape(step.shape[0], -1), axis=1)
    base_norm = np.linalg.norm(base64.reshape(base.shape[0], -1), axis=1)
    scale = np.where(
        step_norm > 1e-12,
        target_relative_l2 * np.maximum(base_norm, 1e-12) / step_norm,
        0.0,
    )
    return (step64 * scale[:, None, None, None]).astype(np.float32)


def _gradient_average(
    rows: Sequence[tuple[list[torch.Tensor | None], int]],
) -> list[torch.Tensor | None] | None:
    if not rows:
        return None
    total = sum(weight for _gradients, weight in rows)
    result: list[torch.Tensor | None] = []
    for position in range(len(rows[0][0])):
        available = [
            (gradients[position], weight)
            for gradients, weight in rows
            if gradients[position] is not None
        ]
        result.append(
            None
            if not available
            else sum(gradient * weight for gradient, weight in available) / total
        )
    return result


def _gradient_cosine(
    left: Sequence[torch.Tensor | None] | None,
    right: Sequence[torch.Tensor | None] | None,
) -> float | None:
    if left is None or right is None:
        return None
    pairs = [
        (a, b)
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    if not pairs:
        return None
    dot = sum((a * b).sum() for a, b in pairs)
    left_norm = sum(a.pow(2).sum() for a, _b in pairs).sqrt()
    right_norm = sum(b.pow(2).sum() for _a, b in pairs).sqrt()
    return float((dot / (left_norm * right_norm + 1e-12)).detach().cpu())


class ProgressiveFeedbackProxy:
    def __init__(
        self,
        args,
        split: dict,
        output_root: Path,
        *,
        retain_payloads_for_replay: bool,
    ):
        self.args = args
        self.mode = args.progressive_proxy_mode
        self.total_tasks = len(split["new_order"])
        self.proxy_tasks = resolve_task_spec(
            args.progressive_proxy_tasks,
            self.total_tasks,
        )
        self.clean_feedback_tasks = resolve_task_spec(
            args.progressive_clean_feedback_tasks,
            self.total_tasks,
        )
        self.proxy_blocks = load_pretrained(args)
        self.optimizer = torch.optim.Adam(
            [parameter for block in self.proxy_blocks for parameter in block.parameters()],
            lr=args.progressive_proxy_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        self.rng = np.random.default_rng(args.seed + 8_300_111)
        source_data: list[Path] = []
        source_labels: list[Path] = []
        for source_subject in split["train_idx"]:
            data_paths, label_paths = subject_paths(args.data_root, source_subject)
            source_data.extend(data_paths)
            source_labels.extend(label_paths)
        self.labeled_records = list(zip(source_data, source_labels))
        self.population_mode = self.mode == "population_feedback"
        if self.population_mode:
            self.base_paths = []
            self.base_label_paths = []
            self.initial_pool = None
            self.current_pool = None
        else:
            base_data, base_labels = subject_paths(
                args.data_root,
                args.progressive_base_subject,
            )
            if not base_data:
                raise FileNotFoundError(
                    f"No fixed proxy data for subject {args.progressive_base_subject}"
                )
            self.base_paths = list(base_data)
            self.base_label_paths = list(base_labels)
            self.initial_pool = _load_arrays(self.base_paths)
            self.current_pool = self.initial_pool.copy()
        active_fraction = getattr(args, "progressive_active_fraction", 1.0)
        if self.population_mode:
            self.active_mask = np.zeros(0, dtype=bool)
        else:
            active_count = max(1, int(math.ceil(active_fraction * len(self.base_paths))))
            if active_count < len(self.base_paths):
                base_probabilities = public_probabilities(
                    self.proxy_blocks,
                    self.base_paths,
                    args,
                )
                confidence = base_probabilities.max(axis=2).mean(axis=1)
                active_indices = np.argsort(confidence, kind="stable")[:active_count]
            else:
                active_indices = np.arange(len(self.base_paths))
            self.active_mask = np.zeros(len(self.base_paths), dtype=bool)
            self.active_mask[active_indices] = True
        self.input_direction: np.ndarray | None = None
        self.persist_state = ProbabilityStateFilter() if getattr(args, "progressive_persist", False) else None
        self.direction_bank = (
            DirectionBank(
                capacity=args.progressive_direction_bank_capacity,
                decay=args.progressive_direction_bank_decay,
            )
            if getattr(args, "progressive_persist", False)
            else None
        )
        self.history_gradients: list[torch.Tensor | None] | None = None
        self.feedback_records: list[dict[str, Any]] = []
        self.task_rows: dict[int, dict[str, Any]] = {}
        self.output_root = output_root / "progressive_proxy"
        self.payload_root = self.output_root / "payload"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.retain_payloads_for_replay = retain_payloads_for_replay
        self.generated_task_dirs: list[Path] = []
        self.population_class_counts: list[int] = []
        self.population_subject_count = 0
        self.population_build_count = 0
        self.population_record_kind_counts: dict[str, int] = {}
        self.population_conflict_map: list[int] = []
        self.population_seed_invariant_drifts: list[float] = []

    def protocol(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "victim_parameters_visible": False,
            "public_feedback": (
                f"post-task {self.args.model_param.NumClasses}-class "
                "probabilities on owned uploads"
            ),
            "source_pretrain_hard_labels_visible": True,
            "incremental_clean_hard_labels_visible": False,
            "clean_feedback_tasks": sorted(self.clean_feedback_tasks),
            "proxy_tasks": sorted(self.proxy_tasks),
            "base_subject": None if self.population_mode else int(self.args.progressive_base_subject),
            "base_sequences": len(self.base_paths) if not self.population_mode else 0,
            "population_mode": self.population_mode,
            "population_refresh_mix": float(getattr(self.args, "progressive_population_refresh_mix", 0.20)),
            "population_class_counts": self.population_class_counts,
            "population_subject_count": int(self.population_subject_count),
            "population_record_kind_counts": self.population_record_kind_counts,
            "population_cross_class_mix": float(
                getattr(self.args, "progressive_population_cross_class_mix", 0.0)
            ),
            "population_conflict_map": self.population_conflict_map,
            "preserve_eeg_invariants": bool(
                getattr(self.args, "progressive_preserve_eeg_invariants", False)
            ),
            "invariant_drift_tolerance": float(
                getattr(self.args, "progressive_invariant_drift_tolerance", 0.02)
            ),
            "population_invariants": [
                "class_distribution",
                "relative_bandpower",
                "channel_covariance_spectrum",
                "temporal_autocorrelation",
            ] if self.population_mode else [],
            "upload_full_pool": bool(
                getattr(self.args, "progressive_upload_full_pool", False)
            ),
            "match_task_sequence_count": bool(
                getattr(
                    self.args,
                    "progressive_match_task_sequence_count",
                    False,
                )
            ),
            "active_sequences": int(self.active_mask.sum()),
            "passive_sequences": int((~self.active_mask).sum()),
            "require_all_sequences_modified": bool(
                getattr(
                    self.args,
                    "progressive_require_all_sequences_modified",
                    False,
                )
            ),
            "task_count_changed": False,
            "subject_order_changed": False,
            "step_relative_l2": self.args.progressive_step_relative_l2,
            "cumulative_relative_l2": self.args.progressive_cumulative_relative_l2,
            "history_weight": self.args.progressive_history_weight,
            "persist_eeg": bool(getattr(self.args, "progressive_persist", False)),
            "persist_state_source": "owned_epoch_probabilities_and_proxy_input_directions_only",
            "direction_bank_capacity": int(getattr(self.args, "progressive_direction_bank_capacity", 4)),
            "source_labeled_sequences": len(self.labeled_records),
            "survival_objective": {
                "enabled": self.args.progressive_survival_weight > 0.0,
                "trajectories": self.args.progressive_survival_trajectories,
                "steps_per_trajectory": self.args.progressive_survival_steps,
                "batch_sequences": self.args.progressive_survival_batch,
                "weight": self.args.progressive_survival_weight,
                "softmin_temperature": self.args.progressive_survival_temperature,
                "repair_data": "source_pretrain_hard_labels_only",
                "future_incremental_data_visible": False,
            },
        }

    def is_proxy_task(self, task_index: int) -> bool:
        return int(task_index) in self.proxy_tasks

    def is_clean_feedback_task(self, task_index: int) -> bool:
        return int(task_index) in self.clean_feedback_tasks

    def _record_distribution(self, labels_or_probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(labels_or_probabilities)
        if values.ndim == 1:
            classes = self.args.model_param.NumClasses
            histogram = np.bincount(values.astype(np.int64), minlength=classes).astype(np.float32)
            return histogram / max(float(histogram.sum()), 1.0)
        probabilities = values.astype(np.float32)
        probabilities = probabilities.reshape(-1, probabilities.shape[-1])
        distribution = probabilities.mean(axis=0)
        return distribution / max(float(distribution.sum()), 1e-8)

    def _population_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for data_path, label_path in self.labeled_records:
            labels = np.load(label_path, allow_pickle=False)
            subject_name = data_path.parent.parent.name
            subject_digits = "".join(character for character in subject_name if character.isdigit())
            records.append({
                "data": data_path,
                "distribution": self._record_distribution(labels),
                "subject": int(subject_digits) if subject_digits else -1,
                "kind": "source",
            })
        if getattr(self.args, "progressive_feedback_weight", 1.0) >= 0.0:
            for row in self.feedback_records:
                records.append({
                    "data": row["data"],
                    "distribution": self._record_distribution(row["probabilities"]),
                    "subject": int(row["subject"]),
                    "kind": str(row["kind"]),
                })
        return records

    def _load_population_record(self, record: dict[str, Any]) -> np.ndarray:
        data = record["data"]
        if isinstance(data, np.ndarray):
            return data.astype(np.float32, copy=False)
        return np.load(data, allow_pickle=False).astype(np.float32, copy=False)

    def _build_population_pool(self, count: int) -> np.ndarray:
        records = self._population_records()
        if not records:
            raise RuntimeError("Population Proxy requires source or feedback records")
        classes = self.args.model_param.NumClasses
        buckets: list[list[int]] = [[] for _ in range(classes)]
        for index, record in enumerate(records):
            buckets[int(np.argmax(record["distribution"]))].append(index)
        nonempty = [index for index, bucket in enumerate(buckets) if bucket]
        if not nonempty:
            raise RuntimeError("Population Proxy has no class-labelled records")
        representative_buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
        invariant_centroids: dict[int, np.ndarray] = {}
        candidates_per_class = int(
            getattr(self.args, "progressive_population_candidates_per_class", 4)
        )
        for class_index in nonempty:
            bucket = buckets[class_index]
            proxy_indices = [index for index in bucket if records[index]["kind"] == "proxy"]
            clean_indices = [index for index in bucket if records[index]["kind"] == "clean"]
            source_indices = [index for index in bucket if records[index]["kind"] == "source"]
            candidate_indices: list[int] = []
            if proxy_indices:
                proxy_quota = min(len(proxy_indices), max(1, candidates_per_class // 2))
                candidate_indices.extend(
                    int(index) for index in self.rng.choice(proxy_indices, size=proxy_quota, replace=False)
                )
            for candidates in (clean_indices, source_indices, bucket):
                remaining = candidates_per_class - len(candidate_indices)
                available = [index for index in candidates if index not in candidate_indices]
                if remaining <= 0 or not available:
                    continue
                candidate_indices.extend(
                    int(index) for index in self.rng.choice(
                        available,
                        size=min(remaining, len(available)),
                        replace=False,
                    )
                )
            candidates = [
                (int(index), self._load_population_record(records[int(index)]))
                for index in candidate_indices
            ]
            descriptors = np.stack(
                [_eeg_invariant_descriptor(array) for _index, array in candidates]
            )
            centroid = descriptors.mean(axis=0)
            invariant_centroids[class_index] = centroid
            distances = np.linalg.norm(descriptors - centroid, axis=1)
            representative_buckets[class_index] = [
                candidates[int(index)] for index in np.argsort(distances, kind="stable")
            ]
        confusion = np.zeros((classes, classes), dtype=np.float64)
        for record in records:
            if record["kind"] == "source":
                continue
            distribution = np.asarray(record["distribution"], dtype=np.float64)
            source_class = int(np.argmax(distribution))
            confusion[source_class] += distribution
        np.fill_diagonal(confusion, 0.0)
        conflict_map: list[int] = []
        for class_index in range(classes):
            available_targets = [target for target in nonempty if target != class_index]
            if not available_targets or class_index not in invariant_centroids:
                conflict_map.append(class_index)
                continue
            feedback_scores = confusion[class_index, available_targets]
            if float(feedback_scores.max()) > 0.0:
                conflict_map.append(available_targets[int(np.argmax(feedback_scores))])
                continue
            source_centroid = invariant_centroids[class_index]
            distances = [
                float(np.linalg.norm(source_centroid - invariant_centroids[target]))
                for target in available_targets
            ]
            conflict_map.append(available_targets[int(np.argmin(distances))])
        self.population_conflict_map = conflict_map
        selected: list[np.ndarray] = []
        selected_classes: list[int] = []
        selected_subjects: set[int] = set()
        selected_kinds: list[str] = []
        class_offset = getattr(self, "population_build_count", 0) % len(nonempty)
        self.population_build_count = getattr(self, "population_build_count", 0) + 1
        for position in range(count):
            class_index = nonempty[(position + class_offset) % len(nonempty)]
            representatives = representative_buckets[class_index]
            first_index, first = representatives[position % len(representatives)]
            first_record = records[first_index]
            cross_class_mix = float(
                getattr(self.args, "progressive_population_cross_class_mix", 0.0)
            )
            second_class = (
                conflict_map[class_index] if cross_class_mix > 0.0 else class_index
            )
            second_representatives = representative_buckets.get(
                second_class,
                representatives,
            )
            second_candidates = [
                (index, array) for index, array in second_representatives
                if records[index]["subject"] != first_record["subject"]
            ] or second_representatives
            second_index, second = second_candidates[
                int(self.rng.integers(0, len(second_candidates)))
            ]
            alpha = (
                float(np.clip(1.0 - cross_class_mix + self.rng.uniform(-0.025, 0.025), 0.51, 0.99))
                if cross_class_mix > 0.0
                else float(self.rng.uniform(0.35, 0.65))
            )
            mixed = _fit_eeg_invariants(first, second, alpha)
            invariant_dataset = getattr(
                self.args,
                "dataset",
                "FACED" if first.shape[1] == 32 else "ISRUC",
            )
            seed_drift = invariant_drift(mixed, first, invariant_dataset)
            if getattr(self.args, "progressive_preserve_eeg_invariants", False):
                mixed, seed_drift = _limit_invariant_drift(
                    mixed,
                    first,
                    invariant_dataset,
                    self.args.progressive_invariant_drift_tolerance,
                )
            selected.append(mixed.astype(np.float32, copy=False))
            if not hasattr(self, "population_seed_invariant_drifts"):
                self.population_seed_invariant_drifts = []
            self.population_seed_invariant_drifts.append(float(seed_drift))
            selected_classes.append(class_index)
            selected_subjects.add(int(first_record["subject"]))
            selected_kinds.append(str(first_record["kind"]))
        self.population_class_counts = [selected_classes.count(index) for index in range(classes)]
        self.population_subject_count = len(selected_subjects)
        self.population_record_kind_counts = {
            kind: selected_kinds.count(kind) for kind in ("source", "clean", "proxy")
        }
        return np.stack(selected, axis=0)

    def _initialize_population_pool(self, count: int) -> None:
        pool = self._build_population_pool(count)
        self.initial_pool = pool.copy()
        self.current_pool = pool.copy()
        self.active_mask = np.ones(count, dtype=bool)
        self.input_direction = None

    def _refresh_population_reference(self) -> None:
        if not self.population_mode or self.current_pool is None:
            return
        current_before = self.current_pool.copy()
        reference = self._build_population_pool(len(self.current_pool))
        mix = float(getattr(self.args, "progressive_population_refresh_mix", 0.20))
        self.current_pool = ((1.0 - mix) * self.current_pool + mix * reference).astype(np.float32)
        if getattr(self.args, "progressive_preserve_eeg_invariants", False):
            self.current_pool = _project_numpy(
                self.current_pool,
                current_before,
                max_relative_l2=self.args.progressive_step_relative_l2,
                max_linf_over_std=self.args.progressive_step_linf_std,
            )
            self.current_pool, _drift = _limit_batch_invariant_drift(
                self.current_pool,
                self.initial_pool,
                self.args.dataset,
                self.args.progressive_invariant_drift_tolerance,
            )

    def _ensure_pool_capacity(self, count: int) -> None:
        if getattr(self, "population_mode", False) and self.current_pool is None:
            self._initialize_population_pool(count)
            return
        if count <= len(self.current_pool):
            return
        original_count = len(self.current_pool)
        if original_count == 0:
            raise ValueError("Cannot expand an empty progressive Proxy pool")
        extra_indices = np.arange(count - original_count) % original_count
        extra_initial = self.initial_pool[extra_indices].copy()
        extra_current = self.current_pool[extra_indices].copy()
        self.initial_pool = np.concatenate((self.initial_pool, extra_initial), axis=0)
        self.current_pool = np.concatenate((self.current_pool, extra_current), axis=0)
        self.active_mask = np.concatenate(
            (self.active_mask, self.active_mask[extra_indices]),
            axis=0,
        )
        if self.input_direction is not None:
            extra_direction = extra_current - extra_initial
            self.input_direction = np.concatenate(
                (self.input_direction, extra_direction),
                axis=0,
            )

    def _array_loader(self, arrays: np.ndarray, *, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.args.seed + 5_000_000 + len(self.task_rows))
        return DataLoader(
            ArrayUnlabeledDataset(arrays),
            batch_size=self.args.batch,
            shuffle=shuffle,
            num_workers=0,
            generator=generator if shuffle else None,
        )

    def _adapt_guide(self, task_index: int, arrays: np.ndarray):
        guide = clone_blocks(self.proxy_blocks, self.args)
        if self.args.progressive_guide_epochs <= 0:
            return guide, []
        probe = CPCProbe(guide, self.args)
        losses: list[float] = []
        for _epoch in range(self.args.progressive_guide_epochs):
            rows: list[float] = []
            for eog, eeg, _dummy in self._array_loader(arrays, shuffle=True):
                loss, guide = probe.update(
                    eeg.to(self.args.device),
                    eog.to(self.args.device),
                )
                rows.append(float(loss))
            losses.append(float(np.mean(rows)))
        return guide, losses

    def _sample_labeled_reference(self, count: int):
        indices = self.rng.choice(
            len(self.labeled_records),
            size=count,
            replace=len(self.labeled_records) < count,
        )
        records = [self.labeled_records[int(index)] for index in indices]
        arrays = _load_arrays([data_path for data_path, _label_path in records])
        labels = np.stack(
            [np.load(label_path, allow_pickle=False) for _data_path, label_path in records]
        ).astype(np.int64)
        eog, eeg = _split_modalities(arrays, self.args.device)
        return eog, eeg, torch.from_numpy(labels).to(self.args.device)

    def _generation_args(self):
        generation_args = copy.copy(self.args)
        generation_args.attack_steps = self.args.progressive_generation_steps
        generation_args.attack_generation_batch = (
            self.args.progressive_generation_batch
        )
        generation_args.attack_reference_batch = self.args.progressive_reference_batch
        generation_args.attack_max_relative_l2 = self.args.progressive_step_relative_l2
        generation_args.attack_eps_scale = self.args.progressive_step_linf_std
        generation_args.attack_param_scope = "classifier"
        generation_args.attack_curvature_scale = 0.0
        generation_args.attack_random_start = False
        generation_args.progressive_history_weight = self.args.progressive_history_weight
        generation_args.attack_survival_weight = (
            self.args.progressive_survival_weight
        )
        generation_args.attack_survival_temperature = (
            self.args.progressive_survival_temperature
        )
        overrides = {
            "attack_target_weight": "progressive_target_weight",
            "attack_conflict_weight": "progressive_conflict_weight",
            "attack_gradient_norm_weight": "progressive_gradient_norm_weight",
            "attack_virtual_old_weight": "progressive_virtual_old_weight",
            "attack_virtual_new_weight": "progressive_virtual_new_weight",
            "attack_confidence_weight": "progressive_confidence_weight",
            "attack_l2_weight": "progressive_l2_weight",
        }
        for attack_name, progressive_name in overrides.items():
            value = getattr(self.args, progressive_name, -1.0)
            if value >= 0.0:
                setattr(generation_args, attack_name, value)
        return generation_args

    def _sample_survival_trajectories(self):
        trajectories = self.args.progressive_survival_trajectories
        steps = self.args.progressive_survival_steps
        if self.args.progressive_survival_weight <= 0.0:
            return None
        return [
            [
                self._sample_labeled_reference(
                    self.args.progressive_survival_batch
                )
                for _step in range(steps)
            ]
            for _trajectory in range(trajectories)
        ]

    def _refresh_history_gradients(self, guide, generation_args) -> None:
        count = getattr(self.args, "progressive_history_refresh_count", 0)
        proxy_records = [
            row for row in self.feedback_records if row["kind"] == "proxy"
        ]
        if count <= 0 or not proxy_records:
            return
        selected = proxy_records[-min(count, len(proxy_records)) :]
        arrays = _load_arrays([row["data"] for row in selected])
        batch_size = self.args.progressive_generation_batch
        rows: list[tuple[list[torch.Tensor | None], int]] = []
        for start in range(0, len(arrays), batch_size):
            batch = arrays[start : start + batch_size]
            eog, eeg = _split_modalities(batch, self.args.device)
            gradients = pseudo_update_gradients(
                self.proxy_blocks,
                guide,
                eog,
                eeg,
                generation_args,
            )
            rows.append((gradients, len(batch)))
        refreshed = _gradient_average(rows)
        if refreshed is not None:
            self.history_gradients = [
                None if value is None else value.detach().cpu()
                for value in refreshed
            ]

    def _source_gradient_average(self, generation_args):
        total = getattr(self.args, "progressive_source_gate_samples", 0)
        if total <= 0:
            total = self.args.progressive_reference_batch
        batch_size = self.args.progressive_reference_batch
        rows: list[tuple[list[torch.Tensor | None], int]] = []
        remaining = total
        while remaining > 0:
            count = min(batch_size, remaining)
            source_eog, source_eeg, source_targets = self._sample_labeled_reference(
                count
            )
            gradients = supervised_update_gradients(
                self.proxy_blocks,
                source_eog,
                source_eeg,
                source_targets,
                generation_args,
            )
            rows.append((gradients, count))
            remaining -= count
        return _gradient_average(rows)

    def _generate_progressive_pool(self, task_index: int):
        self._refresh_population_reference()
        guide, cpc_losses = self._adapt_guide(task_index, self.current_pool)
        generation_args = self._generation_args()
        self._refresh_history_gradients(guide, generation_args)
        survival_trajectories = self._sample_survival_trajectories()
        candidate_batches: list[np.ndarray] = []
        diagnostic_rows: list[tuple[dict[str, float], int]] = []
        history_device = (
            None
            if self.history_gradients is None
            else [
                None if value is None else value.to(self.args.device)
                for value in self.history_gradients
            ]
        )
        generation_batch = self.args.progressive_generation_batch
        for start in range(0, len(self.current_pool), generation_batch):
            arrays = self.current_pool[start : start + generation_batch]
            eog, eeg = _split_modalities(arrays, self.args.device)
            reference_eog, reference_eeg, reference_targets = (
                self._sample_labeled_reference(
                    self.args.progressive_reference_batch
                )
            )
            eog_adv, eeg_adv, diagnostics = proxy_dual_harm_batch(
                self.proxy_blocks,
                guide,
                eog,
                eeg,
                reference_eog,
                reference_eeg,
                generation_args,
                strategy=None,
                reference_targets=reference_targets,
                history_gradients=history_device,
                survival_trajectories=survival_trajectories,
            )
            candidate = torch.cat((eog_adv, eeg_adv), dim=2).cpu().numpy()
            candidate_batches.append(candidate.astype(np.float32, copy=False))
            diagnostic_rows.append((diagnostics, len(arrays)))
        candidate_pool = np.concatenate(candidate_batches, axis=0)
        step = candidate_pool - self.current_pool
        input_cosine = None
        direction_reference = self.input_direction
        if self.direction_bank is not None and self.direction_bank.mean() is not None:
            shared_direction = self.direction_bank.mean()
            direction_reference = np.broadcast_to(
                shared_direction[None, ...],
                step.shape,
            )
        if direction_reference is not None:
            step, input_cosine = _constrain_step_to_direction(
                step,
                direction_reference,
                self.args.progressive_input_cone_residual,
            )
        if getattr(self.args, "progressive_fill_step_budget", False):
            step = _fill_step_relative_l2(
                step,
                self.current_pool,
                self.args.progressive_step_relative_l2,
            )
        passive_mask = ~self.active_mask
        if passive_mask.any():
            step[passive_mask] *= getattr(
                self.args,
                "progressive_passive_step_scale",
                1.0,
            )
        candidate_pool = self.current_pool + step
        candidate_pool = _project_numpy(
            candidate_pool,
            self.current_pool,
            max_relative_l2=self.args.progressive_step_relative_l2,
            max_linf_over_std=self.args.progressive_step_linf_std,
        )
        candidate_pool = _project_numpy(
            candidate_pool,
            self.initial_pool,
            max_relative_l2=self.args.progressive_cumulative_relative_l2,
            max_linf_over_std=self.args.progressive_cumulative_linf_std,
        )
        passive_cumulative_l2 = getattr(
            self.args,
            "progressive_passive_cumulative_relative_l2",
            -1.0,
        )
        passive_cumulative_linf = getattr(
            self.args,
            "progressive_passive_cumulative_linf_std",
            -1.0,
        )
        if passive_mask.any() and passive_cumulative_l2 >= 0.0:
            if passive_cumulative_linf < 0.0:
                passive_cumulative_linf = self.args.progressive_cumulative_linf_std
            candidate_pool[passive_mask] = _project_numpy(
                candidate_pool[passive_mask],
                self.initial_pool[passive_mask],
                max_relative_l2=passive_cumulative_l2,
                max_linf_over_std=passive_cumulative_linf,
            )
        invariant_drifts = None
        if getattr(self.args, "progressive_preserve_eeg_invariants", False):
            for _iteration in range(10):
                candidate_pool = _project_numpy(
                    candidate_pool,
                    self.current_pool,
                    max_relative_l2=self.args.progressive_step_relative_l2,
                    max_linf_over_std=self.args.progressive_step_linf_std,
                )
                candidate_pool = _project_numpy(
                    candidate_pool,
                    self.initial_pool,
                    max_relative_l2=self.args.progressive_cumulative_relative_l2,
                    max_linf_over_std=self.args.progressive_cumulative_linf_std,
                )
                candidate_pool, invariant_drifts = _limit_batch_invariant_drift(
                    candidate_pool,
                    self.initial_pool,
                    self.args.dataset,
                    self.args.progressive_invariant_drift_tolerance,
                )
            final_step = _relative_l2(candidate_pool - self.current_pool, self.current_pool)
            final_cumulative = _relative_l2(candidate_pool - self.initial_pool, self.initial_pool)
            if float(final_step.max()) > self.args.progressive_step_relative_l2 + 1e-3:
                raise RuntimeError("Invariant-preserving Proxy violated the step L2 constraint")
            if float(final_cumulative.max()) > self.args.progressive_cumulative_relative_l2 + 1e-3:
                raise RuntimeError("Invariant-preserving Proxy violated the cumulative L2 constraint")
            if float(invariant_drifts.max()) > self.args.progressive_invariant_drift_tolerance + 1e-5:
                raise RuntimeError("Invariant-preserving Proxy exceeded descriptor tolerance")
        gradient_rows: list[tuple[list[torch.Tensor | None], int]] = []
        for start in range(0, len(candidate_pool), generation_batch):
            arrays = candidate_pool[start : start + generation_batch]
            eog, eeg = _split_modalities(arrays, self.args.device)
            gradients = pseudo_update_gradients(
                self.proxy_blocks,
                guide,
                eog,
                eeg,
                generation_args,
            )
            gradient_rows.append((gradients, len(arrays)))
        current_gradients = _gradient_average(gradient_rows)
        history_cosine = _gradient_cosine(current_gradients, history_device)
        source_gradients = self._source_gradient_average(generation_args)
        source_gradient_cosine = _gradient_cosine(
            current_gradients,
            source_gradients,
        )
        max_source_cosine = getattr(
            self.args,
            "progressive_max_source_gradient_cosine",
            1.0,
        )
        source_conflict_accepted = (
            source_gradient_cosine is not None
            and source_gradient_cosine <= max_source_cosine
        )
        if (
            getattr(self.args, "progressive_require_source_conflict", False)
            and not source_conflict_accepted
        ):
            raise RuntimeError(
                f"Task {task_index} proxy/source gradient cosine "
                f"{source_gradient_cosine} exceeds {max_source_cosine}"
            )
        if self.input_direction is None:
            self.input_direction = candidate_pool - self.initial_pool
        persist_diagnostics = {}
        if self.direction_bank is not None:
            current_direction = candidate_pool - self.initial_pool
            persist_diagnostics = self.direction_bank.update(current_direction)
            persist_diagnostics["direction_bank_cosine"] = self.direction_bank.cosine(current_direction)
        if current_gradients is not None:
            current_cpu = [
                None if value is None else value.detach().cpu()
                for value in current_gradients
            ]
            if self.history_gradients is None:
                self.history_gradients = current_cpu
            else:
                decay = self.args.progressive_history_decay
                self.history_gradients = [
                    None
                    if old is None or new is None
                    else decay * old + (1.0 - decay) * new
                    for old, new in zip(self.history_gradients, current_cpu)
                ]

        total_weight = sum(weight for _row, weight in diagnostic_rows)
        aggregate = {
            key: float(
                sum(row[key] * weight for row, weight in diagnostic_rows)
                / total_weight
            )
            for key in diagnostic_rows[0][0]
        }
        step_delta = candidate_pool - self.current_pool
        cumulative_delta = candidate_pool - self.initial_pool
        step_relative = _relative_l2(step_delta, self.current_pool)
        cumulative_relative = _relative_l2(
            cumulative_delta,
            self.initial_pool,
        )
        modified_mask = cumulative_relative > 1e-8
        if (
            getattr(
                self.args,
                "progressive_require_all_sequences_modified",
                False,
            )
            and not modified_mask.all()
        ):
            raise RuntimeError(
                f"Task {task_index} left "
                f"{int((~modified_mask).sum())}/{len(modified_mask)} uploaded "
                "Proxy sequences unchanged"
            )
        self.current_pool = candidate_pool
        del guide
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "guiding_cpc_losses": cpc_losses,
            "generation": aggregate,
            "mean_step_relative_l2": float(step_relative.mean()),
            "max_step_relative_l2": float(step_relative.max()),
            "mean_cumulative_relative_l2": float(cumulative_relative.mean()),
            "max_cumulative_relative_l2": float(cumulative_relative.max()),
            "min_cumulative_relative_l2": float(cumulative_relative.min()),
            "modified_sequences": int(modified_mask.sum()),
            "unmodified_sequences": int((~modified_mask).sum()),
            "mean_invariant_drift": (
                None if invariant_drifts is None else float(invariant_drifts.mean())
            ),
            "max_invariant_drift": (
                None if invariant_drifts is None else float(invariant_drifts.max())
            ),
            "max_population_seed_invariant_drift": (
                max(self.population_seed_invariant_drifts)
                if self.population_seed_invariant_drifts
                else None
            ),
            "input_direction_cosine": input_cosine,
            "history_gradient_cosine": history_cosine,
            "source_gradient_cosine": source_gradient_cosine,
            "source_conflict_accepted": source_conflict_accepted,
            **persist_diagnostics,
            "mean_active_step_relative_l2": float(
                step_relative[self.active_mask].mean()
            ),
            "mean_active_cumulative_relative_l2": float(
                cumulative_relative[self.active_mask].mean()
            ),
            "mean_passive_step_relative_l2": (
                float(step_relative[passive_mask].mean())
                if passive_mask.any()
                else None
            ),
            "mean_passive_cumulative_relative_l2": (
                float(cumulative_relative[passive_mask].mean())
                if passive_mask.any()
                else None
            ),
        }

    def prepare_task(
        self,
        task_index: int,
        subject: int,
        clean_data_paths: Sequence[Path],
    ) -> tuple[list[Path], set[str], dict[str, Any] | None]:
        if task_index not in self.proxy_tasks:
            self.task_rows[task_index] = {
                "kind": "clean_feedback"
                if task_index in self.clean_feedback_tasks
                else "unobserved_clean",
                "task": int(task_index),
                "subject": int(subject),
            }
            return list(clean_data_paths), set(), None
        count = (
            len(clean_data_paths)
            if getattr(
                self.args,
                "progressive_match_task_sequence_count",
                False,
            )
            else len(self.current_pool)
        )
        self._ensure_pool_capacity(count)
        if self.mode in ("feedback", "population_feedback"):
            last_error = None
            for attempt in range(1, self.args.progressive_generation_attempts + 1):
                try:
                    diagnostics = self._generate_progressive_pool(task_index)
                    diagnostics["generation_attempts"] = attempt
                    break
                except RuntimeError as error:
                    if "proxy/source gradient cosine" not in str(error):
                        raise
                    last_error = error
            else:
                raise RuntimeError(
                    f"Task {task_index} exhausted "
                    f"{self.args.progressive_generation_attempts} source-gated "
                    "candidate attempts"
                ) from last_error
        else:
            self.current_pool = self.initial_pool.copy()
            diagnostics = {
                "guiding_cpc_losses": [],
                "generation": None,
                "mean_step_relative_l2": 0.0,
                "max_step_relative_l2": 0.0,
                "mean_cumulative_relative_l2": 0.0,
                "max_cumulative_relative_l2": 0.0,
                "input_direction_cosine": None,
                "history_gradient_cosine": None,
                "source_gradient_cosine": None,
                "source_conflict_accepted": None,
                "mean_active_step_relative_l2": 0.0,
                "mean_active_cumulative_relative_l2": 0.0,
                "mean_passive_step_relative_l2": 0.0,
                "mean_passive_cumulative_relative_l2": 0.0,
            }
        uploaded_relative = _relative_l2(
            self.current_pool[:count] - self.initial_pool[:count],
            self.initial_pool[:count],
        )
        uploaded_modified = uploaded_relative > 1e-8
        if (
            getattr(
                self.args,
                "progressive_require_all_sequences_modified",
                False,
            )
            and not uploaded_modified.all()
        ):
            raise RuntimeError(
                f"Task {task_index} left "
                f"{int((~uploaded_modified).sum())}/{count} uploaded Proxy "
                "sequences unchanged"
            )
        diagnostics.update(
            {
                "generated_pool_sequences": len(self.current_pool),
                "pool_modified_sequences": diagnostics.get(
                    "modified_sequences",
                    0,
                ),
                "pool_unmodified_sequences": diagnostics.get(
                    "unmodified_sequences",
                    len(self.current_pool),
                ),
                "modified_sequences": int(uploaded_modified.sum()),
                "unmodified_sequences": int((~uploaded_modified).sum()),
                "min_uploaded_cumulative_relative_l2": float(
                    uploaded_relative.min()
                ),
                "mean_uploaded_cumulative_relative_l2": float(
                    uploaded_relative.mean()
                ),
                "max_uploaded_cumulative_relative_l2": float(
                    uploaded_relative.max()
                ),
                "population_class_counts": list(self.population_class_counts),
                "population_subject_count": int(self.population_subject_count),
                "population_record_kind_counts": dict(self.population_record_kind_counts),
                "population_conflict_map": list(self.population_conflict_map),
                "preserve_eeg_invariants": bool(
                    getattr(self.args, "progressive_preserve_eeg_invariants", False)
                ),
            }
        )
        task_dir = self.payload_root / f"task_{task_index}_subject_{subject}"
        task_dir.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []
        for index in range(count):
            path = task_dir / f"{index}.npy"
            np.save(path, self.current_pool[index].astype(np.float32, copy=False))
            output_paths.append(path.resolve())
        self.generated_task_dirs.append(task_dir)
        row = {
            "kind": "progressive_proxy" if self.mode in ("feedback", "population_feedback") else "static_proxy",
            "task": int(task_index),
            "subject": int(subject),
            "proxy_sequences": count,
            "base_subject": None if self.population_mode else int(self.args.progressive_base_subject),
            "population_mode": self.population_mode,
            **diagnostics,
        }
        self.task_rows[task_index] = row
        return output_paths, {str(path) for path in output_paths}, row

    def diagnostic_label_paths(
        self,
        task_index: int,
        clean_label_paths: Sequence[Path],
    ) -> list[Path]:
        if task_index in self.proxy_tasks:
            count = (
                len(clean_label_paths)
                if getattr(
                    self.args,
                    "progressive_match_task_sequence_count",
                    False,
                )
                else len(self.base_label_paths)
            )
            if getattr(self, "population_mode", False):
                return list(clean_label_paths)
            return [
                self.base_label_paths[index % len(self.base_label_paths)]
                for index in range(count)
            ]
        return list(clean_label_paths)

    def _sample_feedback_records(self, count: int) -> list[dict[str, Any]]:
        ages = np.asarray(
            [self.total_tasks - int(row["task"]) for row in self.feedback_records],
            dtype=np.float64,
        )
        weights = np.power(self.args.progressive_feedback_decay, ages)
        weights /= weights.sum()
        indices = self.rng.choice(
            len(self.feedback_records),
            size=count,
            replace=len(self.feedback_records) < count,
            p=weights,
        )
        return [self.feedback_records[int(index)] for index in indices]

    def _feedback_batch(self, records: Sequence[dict[str, Any]]):
        arrays = _load_arrays([row["data"] for row in records])
        probabilities = np.stack([row["probabilities"] for row in records]).astype(
            np.float32
        )
        eog, eeg = _split_modalities(arrays, self.args.device)
        targets = torch.from_numpy(probabilities).to(self.args.device)
        return eog, eeg, targets

    def _labeled_batch(self, count: int):
        indices = self.rng.choice(
            len(self.labeled_records),
            size=count,
            replace=len(self.labeled_records) < count,
        )
        records = [self.labeled_records[int(index)] for index in indices]
        arrays = _load_arrays([row[0] for row in records])
        labels = np.stack(
            [np.load(row[1], allow_pickle=False) for row in records]
        ).astype(np.int64)
        eog, eeg = _split_modalities(arrays, self.args.device)
        return eog, eeg, torch.from_numpy(labels).to(self.args.device)

    @torch.no_grad()
    def _mean_feedback_kl(self, records: Sequence[dict[str, Any]]) -> float:
        if not records:
            return 0.0
        modes = [block.training for block in self.proxy_blocks]
        set_train(self.proxy_blocks, False)
        rows: list[float] = []
        batch_size = self.args.progressive_feedback_batch
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            eog, eeg, targets = self._feedback_batch(batch)
            logits = forward_blocks(self.proxy_blocks, eog, eeg, self.args)
            log_probs = logits.permute(0, 2, 1).log_softmax(dim=2)
            rows.append(
                float(
                    F.kl_div(log_probs, targets, reduction="batchmean").cpu()
                )
            )
        for block, mode in zip(self.proxy_blocks, modes):
            block.train(mode)
        return float(np.mean(rows))

    def _distill(self, current_records: Sequence[dict[str, Any]]) -> dict[str, float]:
        before = self._mean_feedback_kl(current_records)
        losses: list[float] = []
        feedback_losses: list[float] = []
        supervised_losses: list[float] = []
        batch_size = self.args.progressive_feedback_batch
        for _step in range(self.args.progressive_feedback_steps):
            set_train(self.proxy_blocks, True)
            if self.args.freeze_bn_stats:
                freeze_batch_norm_running_stats(self.proxy_blocks)
            source_eog, source_eeg, source_labels = self._labeled_batch(batch_size)
            source_logits = flat_logits(
                forward_blocks(
                    self.proxy_blocks,
                    source_eog,
                    source_eeg,
                    self.args,
                )
            )
            supervised = F.cross_entropy(source_logits, source_labels.reshape(-1))
            feedback = supervised.new_zeros(())
            if self.feedback_records:
                feedback_rows = self._sample_feedback_records(batch_size)
                feedback_eog, feedback_eeg, feedback_targets = self._feedback_batch(
                    feedback_rows
                )
                feedback_logits = forward_blocks(
                    self.proxy_blocks,
                    feedback_eog,
                    feedback_eeg,
                    self.args,
                ).permute(0, 2, 1)
                feedback = F.kl_div(
                    feedback_logits.log_softmax(dim=2),
                    feedback_targets,
                    reduction="batchmean",
                )
            loss = (
                self.args.progressive_source_weight * supervised
                + self.args.progressive_feedback_weight * feedback
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for block in self.proxy_blocks
                    for parameter in block.parameters()
                ],
                self.args.grad_clip,
            )
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            feedback_losses.append(float(feedback.detach().cpu()))
            supervised_losses.append(float(supervised.detach().cpu()))
        after = self._mean_feedback_kl(current_records)
        return {
            "response_kl_before": before,
            "response_kl_after": after,
            "mean_distillation_loss": float(np.mean(losses)) if losses else 0.0,
            "mean_feedback_loss": float(np.mean(feedback_losses)) if losses else 0.0,
            "mean_supervised_loss": float(np.mean(supervised_losses)) if losses else 0.0,
        }

    def observe_task(
        self,
        task_index: int,
        subject: int,
        uploaded_data_paths: Sequence[Path],
        victim_probabilities: np.ndarray,
    ) -> dict[str, Any]:
        visible = task_index in self.proxy_tasks or task_index in self.clean_feedback_tasks
        row = self.task_rows.setdefault(
            task_index,
            {"task": int(task_index), "subject": int(subject)},
        )
        if not visible:
            row["feedback_visible"] = False
            return row
        if len(uploaded_data_paths) != victim_probabilities.shape[0]:
            raise ValueError("Victim response count does not match uploaded sequences")
        kind = "proxy" if task_index in self.proxy_tasks else "clean"
        state_diagnostics = {}
        if self.persist_state is not None:
            state_diagnostics = self.persist_state.update(victim_probabilities)
        current_records: list[dict[str, Any]] = []
        for index, (path, probabilities) in enumerate(
            zip(uploaded_data_paths, victim_probabilities)
        ):
            data: Path | np.ndarray = (
                np.load(path, allow_pickle=False).astype(np.float16)
                if kind == "proxy"
                else Path(path)
            )
            record = {
                "kind": kind,
                "task": int(task_index),
                "subject": int(subject),
                "sequence_index": int(index),
                "data": data,
                "probabilities": probabilities.astype(np.float16),
            }
            self.feedback_records.append(record)
            current_records.append(record)
        if len(self.feedback_records) > self.args.progressive_feedback_capacity:
            self.feedback_records = self.feedback_records[
                -self.args.progressive_feedback_capacity :
            ]
        feedback = (
            self._distill(current_records)
            if self.mode in ("feedback", "population_feedback")
            else {
                "response_kl_before": self._mean_feedback_kl(current_records),
                "response_kl_after": None,
                "mean_distillation_loss": 0.0,
                "mean_feedback_loss": 0.0,
                "mean_supervised_loss": 0.0,
            }
        )
        row.update(
            {
                "feedback_visible": True,
                "feedback_kind": kind,
                "feedback_sequences": len(current_records),
                "labeled_buffer_sequences": len(self.labeled_records),
                "feedback_buffer_sequences": len(self.feedback_records),
                **state_diagnostics,
                **feedback,
            }
        )
        self._save_transcript()
        if kind == "proxy" and not self.retain_payloads_for_replay:
            task_dir = Path(uploaded_data_paths[0]).parent
            shutil.rmtree(task_dir, ignore_errors=True)
        return row

    def _json_task_rows(self) -> list[dict[str, Any]]:
        return [self.task_rows[index] for index in sorted(self.task_rows)]

    def _save_transcript(self) -> None:
        payload = {
            "protocol": self.protocol(),
            "tasks": self._json_task_rows(),
            "buffers": {
                "labeled_sequences": len(self.labeled_records),
                "feedback_sequences": len(self.feedback_records),
                "clean_feedback_sequences": sum(
                    row["kind"] == "clean" for row in self.feedback_records
                ),
                "proxy_feedback_sequences": sum(
                    row["kind"] == "proxy" for row in self.feedback_records
                ),
            },
            "persist_state": self.persist_state.state() if self.persist_state is not None else None,
            "direction_bank": self.direction_bank.state() if self.direction_bank is not None else None,
        }
        (self.output_root / "transcript.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def summary(self) -> dict[str, Any]:
        self._save_transcript()
        proxy_rows = [
            row for row in self.task_rows.values() if row.get("proxy_sequences", 0) > 0
        ]
        return {
            "protocol": self.protocol(),
            "proxy_tasks_completed": len(proxy_rows),
            "clean_feedback_tasks_completed": sum(
                bool(row.get("feedback_visible"))
                and row.get("feedback_kind") == "clean"
                for row in self.task_rows.values()
            ),
            "proxy_sequences_uploaded": sum(
                int(row.get("proxy_sequences", 0)) for row in proxy_rows
            ),
            "final_mean_cumulative_relative_l2": (
                float(proxy_rows[-1]["mean_cumulative_relative_l2"])
                if proxy_rows
                else 0.0
            ),
            "feedback_buffer_sequences": len(self.feedback_records),
            "labeled_buffer_sequences": len(self.labeled_records),
            "transcript": str((self.output_root / "transcript.json").resolve()),
        }

    def cleanup(self) -> None:
        if self.payload_root.is_dir():
            shutil.rmtree(self.payload_root)
