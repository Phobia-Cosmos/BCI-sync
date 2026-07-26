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


def _all_named_parameters(blocks) -> list[tuple[str, nn.Parameter]]:
    block_names = ("feature_extractor", "feature_encoder", "sleep_classifier")
    parameters: list[tuple[str, nn.Parameter]] = []
    for block_name, block in zip(block_names, blocks):
        parameters.extend(
            (f"{block_name}.{name}", parameter)
            for name, parameter in block.named_parameters()
            if parameter.requires_grad
        )
    return parameters


def _attack_named_parameters(
    blocks,
    scope: str,
) -> list[tuple[str, nn.Parameter]]:
    all_parameters = _all_named_parameters(blocks)
    if scope == "classifier":
        return [
            (name, parameter)
            for name, parameter in all_parameters
            if name.startswith("sleep_classifier.")
        ]
    if scope == "encoder_head":
        return [
            (name, parameter)
            for name, parameter in all_parameters
            if name.startswith(("feature_encoder.", "sleep_classifier."))
        ]
    if scope == "all":
        return all_parameters
    raise ValueError(f"Unsupported attack parameter scope: {scope}")


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
    max_relative_l2: float = 0.0,
) -> torch.Tensor:
    delta = delta.clamp(-epsilon, epsilon)
    delta = (base + delta).clamp(value_min, value_max) - base
    if max_relative_l2 > 0:
        flat_delta = delta.reshape(delta.shape[0], -1)
        flat_base = base.reshape(base.shape[0], -1)
        max_norm = (
            torch.linalg.vector_norm(flat_base, dim=1).clamp_min(1e-12)
            * max_relative_l2
        )
        delta_norm = torch.linalg.vector_norm(flat_delta, dim=1).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(delta_norm), max_norm / delta_norm)
        delta = delta * scale.view(-1, 1, 1, 1)
    return delta.detach()


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


def _gradient_norm(gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
    available = [gradient for gradient in gradients if gradient is not None]
    if not available:
        raise RuntimeError("The selected attack parameters produced no gradients")
    return sum(gradient.pow(2).sum() for gradient in available).sqrt()


def _normalized_gradient_sum(
    first: Sequence[torch.Tensor | None],
    second: Sequence[torch.Tensor | None],
    second_weight: float,
) -> list[torch.Tensor | None]:
    first_norm = _gradient_norm(first).clamp_min(1e-12)
    second_norm = _gradient_norm(second).clamp_min(1e-12)
    result: list[torch.Tensor | None] = []
    for first_gradient, second_gradient in zip(first, second):
        if first_gradient is None and second_gradient is None:
            result.append(None)
        elif first_gradient is None:
            result.append(second_weight * second_gradient.detach() / second_norm)
        elif second_gradient is None:
            result.append(first_gradient.detach() / first_norm)
        else:
            result.append(
                first_gradient.detach() / first_norm
                + second_weight * second_gradient.detach() / second_norm
            )
    return result


def _weighted_gradient_cosine(
    candidate: Sequence[torch.Tensor | None],
    target: Sequence[torch.Tensor | None],
    weights: Sequence[torch.Tensor],
) -> torch.Tensor:
    available = [
        (candidate_gradient, target_gradient, weight)
        for candidate_gradient, target_gradient, weight in zip(
            candidate,
            target,
            weights,
        )
        if candidate_gradient is not None and target_gradient is not None
    ]
    if not available:
        raise RuntimeError("The selected attack parameters produced no common gradients")
    dot = sum(
        ((candidate_gradient * weight) * target_gradient).sum()
        for candidate_gradient, target_gradient, weight in available
    )
    candidate_norm = sum(
        (candidate_gradient * weight).pow(2).sum()
        for candidate_gradient, _target_gradient, weight in available
    ).sqrt()
    target_norm = sum(
        target_gradient.pow(2).sum()
        for _candidate_gradient, target_gradient, _weight in available
    ).sqrt()
    return dot / (candidate_norm * target_norm + 1e-12)


def _curvature_bypass_weights(
    strategy,
    all_parameters: Sequence[tuple[str, nn.Parameter]],
    attack_parameters: Sequence[tuple[str, nn.Parameter]],
    scale: float,
) -> list[torch.Tensor]:
    if strategy is None or scale <= 0:
        return [torch.ones_like(parameter) for _name, parameter in attack_parameters]
    curvature = strategy.curvature(all_parameters)
    selected = [curvature[name].detach().abs() for name, _parameter in attack_parameters]
    positive = [values[values > 0] for values in selected if (values > 0).any()]
    if not positive:
        return [torch.ones_like(values) for values in selected]
    reference = torch.cat(positive).median().clamp_min(1e-12)
    return [1.0 / (1.0 + scale * values / reference) for values in selected]


def _classifier_parameter_map(
    attack_parameters: Sequence[tuple[str, nn.Parameter]],
) -> list[tuple[str, nn.Parameter]]:
    prefix = "sleep_classifier."
    if any(not name.startswith(prefix) for name, _parameter in attack_parameters):
        raise ValueError(
            "The differentiable proxy unroll currently requires "
            "--attack-param-scope classifier"
        )
    return [
        (name.removeprefix(prefix), parameter)
        for name, parameter in attack_parameters
    ]


def _proxy_dual_harm_terms(
    *,
    student_blocks,
    label_blocks,
    eog_adv: torch.Tensor,
    eeg_adv: torch.Tensor,
    reference_features: torch.Tensor,
    clean_current_features: torch.Tensor,
    source_labels: torch.Tensor,
    clean_pseudo_labels: torch.Tensor,
    target_labels: torch.Tensor,
    all_parameters: Sequence[tuple[str, nn.Parameter]],
    parameter_tensors: Sequence[nn.Parameter],
    classifier_parameters: Sequence[tuple[str, nn.Parameter]],
    harmful_gradients: Sequence[torch.Tensor | None],
    harmful_norm: torch.Tensor,
    history_gradients: Sequence[torch.Tensor | None] | None,
    curvature_weights: Sequence[torch.Tensor],
    eps_eog: torch.Tensor,
    eps_eeg: torch.Tensor,
    delta_eog: torch.Tensor,
    delta_eeg: torch.Tensor,
    strategy,
    args,
    create_graph: bool,
) -> dict[str, torch.Tensor]:
    """Evaluate the differentiable one-step proxy objective at one upload."""
    guiding_logits = flat_logits(forward_blocks(label_blocks, eog_adv, eeg_adv, args))
    guiding_probs = guiding_logits.softmax(dim=1)
    guiding_confidence = guiding_probs.max(dim=1).values
    actual_pseudo_labels = guiding_logits.detach().argmax(dim=1)
    student_logits = flat_logits(forward_blocks(student_blocks, eog_adv, eeg_adv, args))
    inner_loss = F.cross_entropy(student_logits, actual_pseudo_labels)
    if strategy is not None:
        inner_loss = inner_loss + strategy.penalty(all_parameters)
    update_gradients = torch.autograd.grad(
        inner_loss,
        parameter_tensors,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )
    gradient_conflict = _weighted_gradient_cosine(
        update_gradients,
        harmful_gradients,
        curvature_weights,
    )
    history_alignment = (
        _weighted_gradient_cosine(
            update_gradients,
            history_gradients,
            curvature_weights,
        )
        if history_gradients is not None
        else inner_loss.new_zeros(())
    )
    update_norm = _gradient_norm(update_gradients)
    updated_parameters = {
        name: parameter - args.attack_inner_lr * gradient
        for (name, parameter), gradient in zip(
            classifier_parameters,
            update_gradients,
        )
        if gradient is not None
    }
    virtual_old_logits = flat_logits(
        functional_call(
            student_blocks[2],
            updated_parameters,
            (reference_features,),
            strict=False,
        )
    )
    virtual_new_logits = flat_logits(
        functional_call(
            student_blocks[2],
            updated_parameters,
            (clean_current_features,),
            strict=False,
        )
    )
    virtual_old_loss = F.cross_entropy(virtual_old_logits, source_labels)
    virtual_new_loss = F.cross_entropy(virtual_new_logits, clean_pseudo_labels)
    target_loss = F.cross_entropy(guiding_logits, target_labels)
    confidence_loss = F.relu(args.attack_min_confidence - guiding_confidence).mean()
    l2_loss = (
        delta_eog.pow(2).mean() / eps_eog.pow(2).clamp_min(1e-12)
        + delta_eeg.pow(2).mean() / eps_eeg.pow(2).clamp_min(1e-12)
    )
    objective = (
        args.attack_target_weight * target_loss
        + args.attack_conflict_weight * gradient_conflict
        - getattr(args, "progressive_history_weight", 0.0) * history_alignment
        - args.attack_gradient_norm_weight
        * update_norm
        / harmful_norm.clamp_min(1e-12)
        - args.attack_virtual_old_weight * virtual_old_loss
        - args.attack_virtual_new_weight * virtual_new_loss
        + args.attack_confidence_weight * confidence_loss
        + args.attack_l2_weight * l2_loss
    )
    return {
        "objective": objective,
        "gradient_conflict": gradient_conflict,
        "history_alignment": history_alignment,
        "virtual_old_loss": virtual_old_loss,
        "virtual_new_loss": virtual_new_loss,
        "guiding_logits": guiding_logits,
        "guiding_confidence": guiding_confidence,
    }


def proxy_dual_harm_batch(
    student_blocks,
    label_blocks,
    eog: torch.Tensor,
    eeg: torch.Tensor,
    reference_eog: torch.Tensor,
    reference_eeg: torch.Tensor,
    args,
    *,
    strategy=None,
    reference_targets: torch.Tensor | None = None,
    history_gradients: Sequence[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """State-aware white-box input poisoning for regularization CL.

    A private source proxy supplies an old-loss gradient while clean current
    inputs supply a plasticity gradient. The attack chooses a pseudo-label
    shift that conflicts with both, then differentiates through a one-step
    classifier update to optimize the uploaded EEG/EOG values only.
    """
    if args.attack_param_scope != "classifier":
        raise ValueError(
            "proxy_dual_harm currently requires --attack-param-scope classifier"
        )

    student_modes = [block.training for block in student_blocks]
    label_modes = [block.training for block in label_blocks]
    set_train(student_blocks, False)
    set_train(label_blocks, False)

    eog_base = eog.detach()
    eeg_base = eeg.detach()
    all_parameters = _all_named_parameters(student_blocks)
    attack_parameters = _attack_named_parameters(student_blocks, args.attack_param_scope)
    parameter_tensors = [parameter for _name, parameter in attack_parameters]
    classifier_parameters = _classifier_parameter_map(attack_parameters)
    curvature_weights = _curvature_bypass_weights(
        strategy,
        all_parameters,
        attack_parameters,
        args.attack_curvature_scale,
    )

    with torch.no_grad():
        clean_guiding_logits = flat_logits(
            forward_blocks(label_blocks, eog_base, eeg_base, args)
        )
        clean_pseudo_labels = clean_guiding_logits.argmax(dim=1)
        source_logits = flat_logits(
            forward_blocks(student_blocks, reference_eog, reference_eeg, args)
        )
        source_labels = (
            source_logits.argmax(dim=1)
            if reference_targets is None
            else reference_targets.reshape(-1).long().to(source_logits.device)
        )
        if source_labels.shape[0] != source_logits.shape[0]:
            raise ValueError("Reference target count does not match reference epochs")
        reference_features = _encoded_features(
            student_blocks,
            reference_eog,
            reference_eeg,
            args,
        ).detach()
        clean_current_features = _encoded_features(
            student_blocks,
            eog_base,
            eeg_base,
            args,
        ).detach()

    old_logits = flat_logits(
        forward_blocks(student_blocks, reference_eog, reference_eeg, args)
    )
    old_loss = F.cross_entropy(old_logits, source_labels)
    old_gradients = [
        None if gradient is None else gradient.detach()
        for gradient in torch.autograd.grad(
            old_loss,
            parameter_tensors,
            allow_unused=True,
        )
    ]
    clean_logits = flat_logits(forward_blocks(student_blocks, eog_base, eeg_base, args))
    clean_loss = F.cross_entropy(clean_logits, clean_pseudo_labels)
    clean_gradients = [
        None if gradient is None else gradient.detach()
        for gradient in torch.autograd.grad(
            clean_loss,
            parameter_tensors,
            allow_unused=True,
        )
    ]
    harmful_gradients = _normalized_gradient_sum(
        old_gradients,
        clean_gradients,
        args.attack_new_proxy_weight,
    )
    harmful_norm = _gradient_norm(harmful_gradients).detach()

    class_count = clean_guiding_logits.shape[1]
    candidate_rows: list[tuple[float, int, torch.Tensor]] = []
    for shift in range(1, class_count):
        target_labels = (clean_pseudo_labels + shift) % class_count
        target_logits = flat_logits(
            forward_blocks(student_blocks, eog_base, eeg_base, args)
        )
        target_loss = F.cross_entropy(target_logits, target_labels)
        target_gradients = torch.autograd.grad(
            target_loss,
            parameter_tensors,
            allow_unused=True,
        )
        conflict = _weighted_gradient_cosine(
            target_gradients,
            harmful_gradients,
            curvature_weights,
        )
        history_alignment = (
            _weighted_gradient_cosine(
                target_gradients,
                history_gradients,
                curvature_weights,
            )
            if history_gradients is not None
            else conflict.new_zeros(())
        )
        score = conflict - getattr(
            args, "progressive_history_weight", 0.0
        ) * history_alignment
        candidate_rows.append(
            (float(score.detach().cpu()), shift, target_labels)
        )
    target_conflict, target_shift, target_labels = min(
        candidate_rows,
        key=lambda row: row[0],
    )

    eps_eog, eog_min, eog_max = _bounds(eog_base, args.attack_eps_scale)
    eps_eeg, eeg_min, eeg_max = _bounds(eeg_base, args.attack_eps_scale)
    delta_eog = _initialize_delta(
        eog_base,
        eps_eog,
        eog_min,
        eog_max,
        args.attack_random_start,
    )
    delta_eeg = _initialize_delta(
        eeg_base,
        eps_eeg,
        eeg_min,
        eeg_max,
        args.attack_random_start,
    )
    delta_eog = _project(
        delta_eog,
        eog_base,
        eps_eog,
        eog_min,
        eog_max,
        args.attack_max_relative_l2,
    )
    delta_eeg = _project(
        delta_eeg,
        eeg_base,
        eps_eeg,
        eeg_min,
        eeg_max,
        args.attack_max_relative_l2,
    )
    step_eog = 2.0 * eps_eog / max(args.attack_steps, 1)
    step_eeg = 2.0 * eps_eeg / max(args.attack_steps, 1)

    initial_objective = None
    for _step in range(args.attack_steps):
        delta_eog.requires_grad_(True)
        delta_eeg.requires_grad_(True)
        eog_adv = eog_base + delta_eog
        eeg_adv = eeg_base + delta_eeg
        terms = _proxy_dual_harm_terms(
            student_blocks=student_blocks,
            label_blocks=label_blocks,
            eog_adv=eog_adv,
            eeg_adv=eeg_adv,
            reference_features=reference_features,
            clean_current_features=clean_current_features,
            source_labels=source_labels,
            clean_pseudo_labels=clean_pseudo_labels,
            target_labels=target_labels,
            all_parameters=all_parameters,
            parameter_tensors=parameter_tensors,
            classifier_parameters=classifier_parameters,
            harmful_gradients=harmful_gradients,
            harmful_norm=harmful_norm,
            history_gradients=history_gradients,
            curvature_weights=curvature_weights,
            eps_eog=eps_eog,
            eps_eeg=eps_eeg,
            delta_eog=delta_eog,
            delta_eeg=delta_eeg,
            strategy=strategy,
            args=args,
            create_graph=True,
        )
        if initial_objective is None:
            initial_objective = float(terms["objective"].detach().cpu())
        gradient_eog, gradient_eeg = torch.autograd.grad(
            terms["objective"],
            (delta_eog, delta_eeg),
        )
        delta_eog = _project(
            delta_eog - step_eog * gradient_eog.sign(),
            eog_base,
            eps_eog,
            eog_min,
            eog_max,
            args.attack_max_relative_l2,
        )
        delta_eeg = _project(
            delta_eeg - step_eeg * gradient_eeg.sign(),
            eeg_base,
            eps_eeg,
            eeg_min,
            eeg_max,
            args.attack_max_relative_l2,
        )
    eog_adv = (eog_base + delta_eog).detach()
    eeg_adv = (eeg_base + delta_eeg).detach()
    final_terms = _proxy_dual_harm_terms(
        student_blocks=student_blocks,
        label_blocks=label_blocks,
        eog_adv=eog_adv,
        eeg_adv=eeg_adv,
        reference_features=reference_features,
        clean_current_features=clean_current_features,
        source_labels=source_labels,
        clean_pseudo_labels=clean_pseudo_labels,
        target_labels=target_labels,
        all_parameters=all_parameters,
        parameter_tensors=parameter_tensors,
        classifier_parameters=classifier_parameters,
        harmful_gradients=harmful_gradients,
        harmful_norm=harmful_norm,
        history_gradients=history_gradients,
        curvature_weights=curvature_weights,
        eps_eog=eps_eog,
        eps_eeg=eps_eeg,
        delta_eog=delta_eog,
        delta_eeg=delta_eeg,
        strategy=strategy,
        args=args,
        create_graph=False,
    )
    with torch.no_grad():
        final_guiding_labels = final_terms["guiding_logits"].argmax(dim=1)
        pseudo_preservation = (final_guiding_labels == clean_pseudo_labels).float().mean()
        target_hit_rate = (final_guiding_labels == target_labels).float().mean()
    diagnostics = _perturbation_metrics(
        eog_base,
        eeg_base,
        eog_adv,
        eeg_adv,
        eps_eog,
        eps_eeg,
    )
    diagnostics.update(
        {
            "objective_initial": float(initial_objective or 0.0),
            "objective_final": float(final_terms["objective"].detach().cpu()),
            "gradient_conflict_final": float(
                final_terms["gradient_conflict"].detach().cpu()
            ),
            "history_alignment_final": float(
                final_terms["history_alignment"].detach().cpu()
            ),
            "virtual_old_loss_final": float(
                final_terms["virtual_old_loss"].detach().cpu()
            ),
            "virtual_new_loss_final": float(
                final_terms["virtual_new_loss"].detach().cpu()
            ),
            "target_shift": float(target_shift),
            "target_shift_conflict": float(target_conflict),
            "target_hit_rate": float(target_hit_rate.cpu()),
            "pseudo_label_preservation": float(pseudo_preservation.cpu()),
            "guiding_confidence": float(
                final_terms["guiding_confidence"].mean().detach().cpu()
            ),
        }
    )
    for block, mode in zip(student_blocks, student_modes):
        block.train(mode)
    for block, mode in zip(label_blocks, label_modes):
        block.train(mode)
    return eog_adv, eeg_adv, diagnostics


def pseudo_update_gradients(
    student_blocks,
    label_blocks,
    eog: torch.Tensor,
    eeg: torch.Tensor,
    args,
) -> list[torch.Tensor | None]:
    """Return the local classifier update gradient induced by one upload batch."""
    student_modes = [block.training for block in student_blocks]
    label_modes = [block.training for block in label_blocks]
    set_train(student_blocks, False)
    set_train(label_blocks, False)
    attack_parameters = _attack_named_parameters(
        student_blocks,
        args.attack_param_scope,
    )
    parameter_tensors = [parameter for _name, parameter in attack_parameters]
    with torch.no_grad():
        pseudo = flat_logits(
            forward_blocks(label_blocks, eog, eeg, args)
        ).argmax(dim=1)
    logits = flat_logits(forward_blocks(student_blocks, eog, eeg, args))
    loss = F.cross_entropy(logits, pseudo)
    gradients = [
        None if gradient is None else gradient.detach()
        for gradient in torch.autograd.grad(
            loss,
            parameter_tensors,
            allow_unused=True,
        )
    ]
    for block, mode in zip(student_blocks, student_modes):
        block.train(mode)
    for block, mode in zip(label_blocks, label_modes):
        block.train(mode)
    return gradients


def supervised_update_gradients(
    student_blocks,
    eog: torch.Tensor,
    eeg: torch.Tensor,
    targets: torch.Tensor,
    args,
) -> list[torch.Tensor | None]:
    """Return classifier gradients for locally owned hard-labeled source data."""
    student_modes = [block.training for block in student_blocks]
    set_train(student_blocks, False)
    attack_parameters = _attack_named_parameters(
        student_blocks,
        args.attack_param_scope,
    )
    parameter_tensors = [parameter for _name, parameter in attack_parameters]
    logits = flat_logits(forward_blocks(student_blocks, eog, eeg, args))
    labels = targets.reshape(-1).long().to(logits.device)
    if labels.shape[0] != logits.shape[0]:
        raise ValueError("Source target count does not match source epochs")
    loss = F.cross_entropy(logits, labels)
    gradients = [
        None if gradient is None else gradient.detach()
        for gradient in torch.autograd.grad(
            loss,
            parameter_tensors,
            allow_unused=True,
        )
    ]
    for block, mode in zip(student_blocks, student_modes):
        block.train(mode)
    return gradients


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


def materialize_batched_proxy_dual_harm_subject(
    *,
    student_blocks,
    label_blocks,
    strategy,
    current_data_paths: Sequence[Path],
    reference_data_paths: Sequence[Path],
    output_dir: Path,
    task_index: int,
    subject: int,
    args,
) -> tuple[list[Path], dict]:
    """Materialize the state-aware attack in batches, not one sequence at a time."""
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
    row_weights: list[int] = []
    generation_batch = max(1, int(args.attack_generation_batch))

    for batch_number, start in enumerate(range(0, len(poison_indices), generation_batch)):
        indices = poison_indices[start:start + generation_batch]
        eog, eeg = _stack_sequences(
            [current_data_paths[index] for index in indices],
            args.device,
        )
        reference_start = (
            args.seed + 37 * task_index + batch_number * args.attack_reference_batch
        ) % len(reference_data_paths)
        reference_indices = [
            (reference_start + offset) % len(reference_data_paths)
            for offset in range(args.attack_reference_batch)
        ]
        reference_eog, reference_eeg = _stack_sequences(
            [reference_data_paths[index] for index in reference_indices],
            args.device,
        )
        eog_adv, eeg_adv, diagnostics = proxy_dual_harm_batch(
            student_blocks,
            label_blocks,
            eog,
            eeg,
            reference_eog,
            reference_eeg,
            args,
            strategy=strategy,
        )
        for row, index in enumerate(indices):
            poisoned_path = output_dir / f"{index}.npy"
            poisoned = torch.cat((eog_adv[row], eeg_adv[row]), dim=1)
            np.save(poisoned_path, poisoned.cpu().numpy().astype(np.float32))
            mixed_paths[index] = poisoned_path
        rows.append(diagnostics)
        row_weights.append(len(indices))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate: dict[str, float] = {}
    batch_macro_aggregate: dict[str, float] = {}
    if rows:
        for key in rows[0]:
            values = [row[key] for row in rows]
            aggregate[key] = float(np.average(values, weights=row_weights))
            batch_macro_aggregate[key] = float(np.mean(values))
    return mixed_paths, {
        "mode": "proxy_dual_harm",
        "task": int(task_index),
        "subject": int(subject),
        "poisoned_sequences": int(poison_count),
        "total_sequences": int(total),
        "poison_fraction": float(poison_count / max(total, 1)),
        "poison_indices": poison_indices,
        "eps_scale_of_modality_std": float(args.attack_eps_scale),
        "max_relative_l2": float(args.attack_max_relative_l2),
        "steps": int(args.attack_steps),
        "generation_batch": generation_batch,
        "parameter_scope": args.attack_param_scope,
        "reference": "source-training inputs with victim hard pseudo labels",
        "learner_replay": False,
        "regularizer_state_visible_to_attacker": strategy is not None,
        "generated_inputs": True,
        "diagnostics_aggregation": "sequence_weighted",
        "generation_batch_sizes": row_weights,
        "diagnostics_mean": aggregate,
        "diagnostics_batch_macro_mean": batch_macro_aggregate,
    }
