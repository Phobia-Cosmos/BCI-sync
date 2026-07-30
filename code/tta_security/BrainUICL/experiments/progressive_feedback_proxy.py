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


PROGRESSIVE_PROXY_MODES = ("none", "static", "feedback")


def add_progressive_proxy_args(parser: argparse.ArgumentParser) -> None:
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


def validate_progressive_proxy_args(args, total_tasks: int) -> None:
    if args.progressive_proxy_mode == "none":
        return
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
    if args.progressive_proxy_mode == "feedback":
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
        self.history_gradients: list[torch.Tensor | None] | None = None
        self.feedback_records: list[dict[str, Any]] = []
        self.task_rows: dict[int, dict[str, Any]] = {}
        self.output_root = output_root / "progressive_proxy"
        self.payload_root = self.output_root / "payload"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.retain_payloads_for_replay = retain_payloads_for_replay
        self.generated_task_dirs: list[Path] = []

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
            "base_subject": int(self.args.progressive_base_subject),
            "base_sequences": len(self.base_paths),
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
            "source_labeled_sequences": len(self.labeled_records),
        }

    def is_proxy_task(self, task_index: int) -> bool:
        return int(task_index) in self.proxy_tasks

    def is_clean_feedback_task(self, task_index: int) -> bool:
        return int(task_index) in self.clean_feedback_tasks

    def _ensure_pool_capacity(self, count: int) -> None:
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
        guide, cpc_losses = self._adapt_guide(task_index, self.current_pool)
        generation_args = self._generation_args()
        self._refresh_history_gradients(guide, generation_args)
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
            )
            candidate = torch.cat((eog_adv, eeg_adv), dim=2).cpu().numpy()
            candidate_batches.append(candidate.astype(np.float32, copy=False))
            diagnostic_rows.append((diagnostics, len(arrays)))
        candidate_pool = np.concatenate(candidate_batches, axis=0)
        step = candidate_pool - self.current_pool
        input_cosine = None
        if self.input_direction is not None:
            step, input_cosine = _constrain_step_to_direction(
                step,
                self.input_direction,
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
            "input_direction_cosine": input_cosine,
            "history_gradient_cosine": history_cosine,
            "source_gradient_cosine": source_gradient_cosine,
            "source_conflict_accepted": source_conflict_accepted,
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
        if self.mode == "feedback":
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
            "kind": "progressive_proxy" if self.mode == "feedback" else "static_proxy",
            "task": int(task_index),
            "subject": int(subject),
            "proxy_sequences": count,
            "base_subject": int(self.args.progressive_base_subject),
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
            if self.mode == "feedback"
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
