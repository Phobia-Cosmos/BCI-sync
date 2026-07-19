"""Clean-label poisoning attacks for the regularization-only EEG CL runner.

The attacks use model-generated hard labels only. Historical proxy inputs are
available to the white-box attacker but are never passed to the learner's
optimizer, so they do not constitute replay.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

try:
    from .rttdp_brainuicl_full import flat_logits, forward_blocks, set_train
except ImportError:  # Direct execution through experiments/regularization_cl_eeg.py.
    from rttdp_brainuicl_full import flat_logits, forward_blocks, set_train


TensorPair = tuple[torch.Tensor, torch.Tensor]


def _load_sequence(path: Path, device: torch.device) -> TensorPair:
    sequence = torch.from_numpy(np.load(path).astype(np.float32)).to(device)
    return sequence[:, :2, :].unsqueeze(0), sequence[:, 2:, :].unsqueeze(0)


def _stack_sequences(paths: Sequence[Path], device: torch.device) -> TensorPair:
    sequences = [torch.from_numpy(np.load(path).astype(np.float32)) for path in paths]
    batch = torch.stack(sequences).to(device)
    return batch[:, :, :2, :], batch[:, :, 2:, :]


def _attack_parameters(blocks, scope: str) -> list[nn.Parameter]:
    if scope == "classifier":
        selected = blocks[2:]
    elif scope == "encoder_head":
        selected = blocks[1:]
    elif scope == "all":
        selected = blocks
    else:
        raise ValueError(f"Unsupported attack parameter scope: {scope}")
    return [parameter for block in selected for parameter in block.parameters()]


def _gradient_cosine(
    candidate: Sequence[torch.Tensor | None],
    target: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    available = [
        (candidate_grad, target_grad)
        for candidate_grad, target_grad in zip(candidate, target)
        if candidate_grad is not None and target_grad is not None
    ]
    if not available:
        raise RuntimeError("The selected attack parameters produced no gradients")
    dot = sum((left * right).sum() for left, right in available)
    left_norm = sum(left.pow(2).sum() for left, _right in available).sqrt()
    right_norm = sum(right.pow(2).sum() for _left, right in available).sqrt()
    return dot / (left_norm * right_norm + 1e-12)


def _bounds(tensor: torch.Tensor, eps_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    epsilon = tensor.detach().std(unbiased=False).clamp_min(1e-12) * eps_scale
    return epsilon, tensor.detach().amin(), tensor.detach().amax()


def _project(
    delta: torch.Tensor,
    base: torch.Tensor,
    epsilon: torch.Tensor,
    value_min: torch.Tensor,
    value_max: torch.Tensor,
) -> torch.Tensor:
    delta = delta.clamp(-epsilon, epsilon)
    return ((base + delta).clamp(value_min, value_max) - base).detach()


def _initialize_delta(
    base: torch.Tensor,
    epsilon: torch.Tensor,
    value_min: torch.Tensor,
    value_max: torch.Tensor,
    random_start: bool,
) -> torch.Tensor:
    delta = torch.zeros_like(base)
    if random_start:
        delta.uniform_(-float(epsilon), float(epsilon))
    return _project(delta, base, epsilon, value_min, value_max)


def _perturbation_metrics(
    eog: torch.Tensor,
    eeg: torch.Tensor,
    eog_adv: torch.Tensor,
    eeg_adv: torch.Tensor,
    eps_eog: torch.Tensor,
    eps_eeg: torch.Tensor,
) -> dict[str, float]:
    delta_eog = eog_adv - eog
    delta_eeg = eeg_adv - eeg

    def relative_l2(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        delta_flat = delta.reshape(delta.shape[0], -1)
        base_flat = base.reshape(base.shape[0], -1)
        return torch.linalg.vector_norm(delta_flat, dim=1) / (
            torch.linalg.vector_norm(base_flat, dim=1) + 1e-12
        )

    return {
        "epsilon_eog": float(eps_eog.detach().cpu()),
        "epsilon_eeg": float(eps_eeg.detach().cpu()),
        "linf_eog": float(delta_eog.abs().max().detach().cpu()),
        "linf_eeg": float(delta_eeg.abs().max().detach().cpu()),
        "relative_l2_eog": float(relative_l2(delta_eog, eog).mean().detach().cpu()),
        "relative_l2_eeg": float(relative_l2(delta_eeg, eeg).mean().detach().cpu()),
    }


def pacol_gradient_matching_batch(
    student_blocks,
    label_blocks,
    eog: torch.Tensor,
    eeg: torch.Tensor,
    reference_eog: torch.Tensor,
    reference_eeg: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """PACOL-style white-box gradient matching with hard pseudo labels."""
    student_modes = [block.training for block in student_blocks]
    label_modes = [block.training for block in label_blocks]
    set_train(student_blocks, False)
    set_train(label_blocks, False)
    parameters = _attack_parameters(student_blocks, args.attack_param_scope)

    with torch.no_grad():
        clean_logits = flat_logits(forward_blocks(label_blocks, eog, eeg, args))
        current_labels = clean_logits.argmax(dim=1)
        reference_logits = flat_logits(
            forward_blocks(student_blocks, reference_eog, reference_eeg, args)
        )
        reference_labels = reference_logits.argmax(dim=1)
        flipped_reference_labels = (reference_labels + 1) % reference_logits.shape[1]

    target_logits = flat_logits(
        forward_blocks(student_blocks, reference_eog, reference_eeg, args)
    )
    target_loss = F.cross_entropy(target_logits, flipped_reference_labels)
    target_gradients = torch.autograd.grad(
        target_loss,
        parameters,
        allow_unused=True,
    )
    target_gradients = [
        None if gradient is None else gradient.detach()
        for gradient in target_gradients
    ]

    eps_eog, eog_min, eog_max = _bounds(eog, args.attack_eps_scale)
    eps_eeg, eeg_min, eeg_max = _bounds(eeg, args.attack_eps_scale)
    delta_eog = _initialize_delta(
        eog, eps_eog, eog_min, eog_max, args.attack_random_start
    )
    delta_eeg = _initialize_delta(
        eeg, eps_eeg, eeg_min, eeg_max, args.attack_random_start
    )
    step_eog = 2.0 * eps_eog / max(args.attack_steps, 1)
    step_eeg = 2.0 * eps_eeg / max(args.attack_steps, 1)
    initial_distance = None
    final_distance = None

    for _step in range(args.attack_steps):
        delta_eog.requires_grad_(True)
        delta_eeg.requires_grad_(True)
        candidate_logits = flat_logits(
            forward_blocks(student_blocks, eog + delta_eog, eeg + delta_eeg, args)
        )
        candidate_loss = F.cross_entropy(candidate_logits, current_labels)
        candidate_gradients = torch.autograd.grad(
            candidate_loss,
            parameters,
            create_graph=True,
            allow_unused=True,
        )
        distance = 1.0 - _gradient_cosine(candidate_gradients, target_gradients)
        if initial_distance is None:
            initial_distance = float(distance.detach().cpu())
        gradient_eog, gradient_eeg = torch.autograd.grad(
            distance,
            (delta_eog, delta_eeg),
        )
        delta_eog = _project(
            delta_eog - step_eog * gradient_eog.sign(),
            eog,
            eps_eog,
            eog_min,
            eog_max,
        )
        delta_eeg = _project(
            delta_eeg - step_eeg * gradient_eeg.sign(),
            eeg,
            eps_eeg,
            eeg_min,
            eeg_max,
        )
        final_distance = float(distance.detach().cpu())

    eog_adv = (eog + delta_eog).detach()
    eeg_adv = (eeg + delta_eeg).detach()
    final_candidate_logits = flat_logits(
        forward_blocks(student_blocks, eog_adv, eeg_adv, args)
    )
    final_candidate_loss = F.cross_entropy(final_candidate_logits, current_labels)
    final_candidate_gradients = torch.autograd.grad(
        final_candidate_loss,
        parameters,
        allow_unused=True,
    )
    final_distance = float(
        (1.0 - _gradient_cosine(final_candidate_gradients, target_gradients))
        .detach()
        .cpu()
    )
    with torch.no_grad():
        attacked_labels = flat_logits(
            forward_blocks(label_blocks, eog_adv, eeg_adv, args)
        ).argmax(dim=1)
        preservation = (attacked_labels == current_labels).float().mean()
    diagnostics = _perturbation_metrics(
        eog, eeg, eog_adv, eeg_adv, eps_eog, eps_eeg
    )
    diagnostics.update(
        {
            "objective_initial": float(initial_distance or 0.0),
            "objective_final": float(final_distance or 0.0),
            "pseudo_label_preservation": float(preservation.cpu()),
        }
    )
    for block, mode in zip(student_blocks, student_modes):
        block.train(mode)
    for block, mode in zip(label_blocks, label_modes):
        block.train(mode)
    return eog_adv, eeg_adv, diagnostics


def _encoded_features(blocks, eog: torch.Tensor, eeg: torch.Tensor, args) -> torch.Tensor:
    batch = eeg.shape[0]
    eog_flat = eog.reshape(
        -1,
        args.model_param.EogNum,
        args.model_param.EpochLength,
    )
    eeg_flat = eeg.reshape(
        -1,
        args.model_param.EegNum,
        args.model_param.EpochLength,
    )
    features = blocks[0](eeg_flat, eog_flat)
    features = blocks[1](features)
    return features.reshape(batch, args.model_param.SeqLength, -1)


def brainwash_one_step_batch(
    student_blocks,
    label_blocks,
    eog: torch.Tensor,
    eeg: torch.Tensor,
    reference_eog: torch.Tensor,
    reference_eeg: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """One-step BrainWash approximation over the classifier parameters."""
    if args.attack_param_scope != "classifier":
        raise ValueError("BrainWash currently supports --attack-param-scope classifier")
    student_modes = [block.training for block in student_blocks]
    label_modes = [block.training for block in label_blocks]
    set_train(student_blocks, False)
    set_train(label_blocks, False)
    classifier_parameters = list(student_blocks[2].named_parameters())
    parameter_tensors = [parameter for _name, parameter in classifier_parameters]

    with torch.no_grad():
        current_labels = flat_logits(
            forward_blocks(label_blocks, eog, eeg, args)
        ).argmax(dim=1)
        reference_labels = flat_logits(
            forward_blocks(student_blocks, reference_eog, reference_eeg, args)
        ).argmax(dim=1)
        reference_features = _encoded_features(
            student_blocks, reference_eog, reference_eeg, args
        ).detach()
        clean_current_features = _encoded_features(
            student_blocks, eog, eeg, args
        ).detach()

    eps_eog, eog_min, eog_max = _bounds(eog, args.attack_eps_scale)
    eps_eeg, eeg_min, eeg_max = _bounds(eeg, args.attack_eps_scale)
    delta_eog = _initialize_delta(
        eog, eps_eog, eog_min, eog_max, args.attack_random_start
    )
    delta_eeg = _initialize_delta(
        eeg, eps_eeg, eeg_min, eeg_max, args.attack_random_start
    )
    step_eog = 2.0 * eps_eog / max(args.attack_steps, 1)
    step_eeg = 2.0 * eps_eeg / max(args.attack_steps, 1)
    initial_objective = None
    final_objective = None
    final_old_loss = None
    final_current_loss = None

    for _step in range(args.attack_steps):
        delta_eog.requires_grad_(True)
        delta_eeg.requires_grad_(True)
        attacked_features = _encoded_features(
            student_blocks,
            eog + delta_eog,
            eeg + delta_eeg,
            args,
        )
        attacked_logits = flat_logits(student_blocks[2](attacked_features))
        inner_loss = F.cross_entropy(attacked_logits, current_labels)
        inner_gradients = torch.autograd.grad(
            inner_loss,
            parameter_tensors,
            create_graph=True,
        )
        updated_parameters = {
            name: parameter - args.attack_inner_lr * gradient
            for (name, parameter), gradient in zip(
                classifier_parameters,
                inner_gradients,
            )
        }

        reference_logits = flat_logits(
            functional_call(
                student_blocks[2],
                updated_parameters,
                (reference_features,),
                strict=False,
            )
        )
        old_loss = F.cross_entropy(reference_logits, reference_labels)
        current_logits = flat_logits(
            functional_call(
                student_blocks[2],
                updated_parameters,
                (clean_current_features,),
                strict=False,
            )
        )
        current_loss = F.cross_entropy(current_logits, current_labels)
        objective = -old_loss
        if args.attack_mode == "brainwash_cautious":
            objective = objective + args.attack_cautious_weight * current_loss
        if initial_objective is None:
            initial_objective = float(objective.detach().cpu())
        gradient_eog, gradient_eeg = torch.autograd.grad(
            objective,
            (delta_eog, delta_eeg),
        )
        delta_eog = _project(
            delta_eog - step_eog * gradient_eog.sign(),
            eog,
            eps_eog,
            eog_min,
            eog_max,
        )
        delta_eeg = _project(
            delta_eeg - step_eeg * gradient_eeg.sign(),
            eeg,
            eps_eeg,
            eeg_min,
            eeg_max,
        )
        final_objective = float(objective.detach().cpu())
        final_old_loss = float(old_loss.detach().cpu())
        final_current_loss = float(current_loss.detach().cpu())

    eog_adv = (eog + delta_eog).detach()
    eeg_adv = (eeg + delta_eeg).detach()
    final_attacked_features = _encoded_features(
        student_blocks,
        eog_adv,
        eeg_adv,
        args,
    )
    final_attacked_logits = flat_logits(student_blocks[2](final_attacked_features))
    final_inner_loss = F.cross_entropy(final_attacked_logits, current_labels)
    final_inner_gradients = torch.autograd.grad(
        final_inner_loss,
        parameter_tensors,
    )
    final_updated_parameters = {
        name: parameter - args.attack_inner_lr * gradient
        for (name, parameter), gradient in zip(
            classifier_parameters,
            final_inner_gradients,
        )
    }
    final_reference_logits = flat_logits(
        functional_call(
            student_blocks[2],
            final_updated_parameters,
            (reference_features,),
            strict=False,
        )
    )
    final_current_logits = flat_logits(
        functional_call(
            student_blocks[2],
            final_updated_parameters,
            (clean_current_features,),
            strict=False,
        )
    )
    final_old_loss_tensor = F.cross_entropy(
        final_reference_logits,
        reference_labels,
    )
    final_current_loss_tensor = F.cross_entropy(
        final_current_logits,
        current_labels,
    )
    final_objective_tensor = -final_old_loss_tensor
    if args.attack_mode == "brainwash_cautious":
        final_objective_tensor = (
            final_objective_tensor
            + args.attack_cautious_weight * final_current_loss_tensor
        )
    final_objective = float(final_objective_tensor.detach().cpu())
    final_old_loss = float(final_old_loss_tensor.detach().cpu())
    final_current_loss = float(final_current_loss_tensor.detach().cpu())
    with torch.no_grad():
        attacked_labels = flat_logits(
            forward_blocks(label_blocks, eog_adv, eeg_adv, args)
        ).argmax(dim=1)
        preservation = (attacked_labels == current_labels).float().mean()
    diagnostics = _perturbation_metrics(
        eog, eeg, eog_adv, eeg_adv, eps_eog, eps_eeg
    )
    diagnostics.update(
        {
            "objective_initial": float(initial_objective or 0.0),
            "objective_final": float(final_objective or 0.0),
            "surrogate_old_loss_final": float(final_old_loss or 0.0),
            "surrogate_current_loss_final": float(final_current_loss or 0.0),
            "pseudo_label_preservation": float(preservation.cpu()),
        }
    )
    for block, mode in zip(student_blocks, student_modes):
        block.train(mode)
    for block, mode in zip(label_blocks, label_modes):
        block.train(mode)
    return eog_adv, eeg_adv, diagnostics


def materialize_poisoned_subject(
    *,
    attack: Callable,
    student_blocks,
    label_blocks,
    current_data_paths: Sequence[Path],
    reference_data_paths: Sequence[Path],
    output_dir: Path,
    task_index: int,
    subject: int,
    args,
) -> tuple[list[Path], dict]:
    """Save only selected poisoned sequences and return a mixed path list."""
    total = len(current_data_paths)
    poison_count = int(math.ceil(total * args.attack_fraction))
    poison_count = min(max(poison_count, 0), total)
    rng = np.random.default_rng(args.seed + 1009 * task_index + 9173 * subject)
    poison_indices = sorted(
        rng.choice(total, poison_count, replace=False).astype(int).tolist()
    ) if poison_count else []
    mixed_paths = list(current_data_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    for attack_order, index in enumerate(poison_indices):
        eog, eeg = _load_sequence(current_data_paths[index], args.device)
        reference_start = (
            args.seed + 37 * task_index + attack_order * args.attack_reference_batch
        ) % len(reference_data_paths)
        reference_indices = [
            (reference_start + offset) % len(reference_data_paths)
            for offset in range(args.attack_reference_batch)
        ]
        reference_paths = [reference_data_paths[item] for item in reference_indices]
        reference_eog, reference_eeg = _stack_sequences(
            reference_paths,
            args.device,
        )
        eog_adv, eeg_adv, diagnostics = attack(
            student_blocks,
            label_blocks,
            eog,
            eeg,
            reference_eog,
            reference_eeg,
            args,
        )
        poisoned_path = output_dir / f"{index}.npy"
        poisoned = torch.cat((eog_adv[0], eeg_adv[0]), dim=1)
        np.save(poisoned_path, poisoned.cpu().numpy().astype(np.float32))
        mixed_paths[index] = poisoned_path
        rows.append(diagnostics)

    aggregate: dict[str, float] = {}
    if rows:
        for key in rows[0]:
            aggregate[key] = float(np.mean([row[key] for row in rows]))
    return mixed_paths, {
        "mode": args.attack_mode,
        "task": int(task_index),
        "subject": int(subject),
        "poisoned_sequences": int(poison_count),
        "total_sequences": int(total),
        "poison_fraction": float(poison_count / max(total, 1)),
        "poison_indices": poison_indices,
        "eps_scale_of_modality_std": float(args.attack_eps_scale),
        "steps": int(args.attack_steps),
        "parameter_scope": args.attack_param_scope,
        "reference": "source-training inputs with victim hard pseudo labels",
        "learner_replay": False,
        "diagnostics_mean": aggregate,
    }
