from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


BLOCK_NAMES = ("feature_extractor", "feature_encoder", "sleep_classifier")


def named_trainable_parameters(
    blocks: Sequence[nn.Module],
) -> list[tuple[str, nn.Parameter]]:
    parameters: list[tuple[str, nn.Parameter]] = []
    for block_name, block in zip(BLOCK_NAMES, blocks):
        for name, parameter in block.named_parameters():
            if parameter.requires_grad:
                parameters.append((f"{block_name}.{name}", parameter))
    return parameters


def hard_pseudo_label_loss(
    student_logits: torch.Tensor,
    guiding_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use every guiding-model prediction; no confidence filtering is applied."""
    pseudo_labels = guiding_logits.detach().argmax(dim=1)
    return F.cross_entropy(student_logits, pseudo_labels), pseudo_labels


def freeze_batch_norm_running_stats(blocks: Sequence[nn.Module]) -> int:
    """Use stored BN statistics while leaving affine parameters trainable."""
    frozen = 0
    for block in blocks:
        for module in block.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                frozen += 1
    return frozen


def _clone_parameters(
    parameters: Iterable[tuple[str, nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in parameters}


def _zero_for(parameters: Sequence[tuple[str, nn.Parameter]]) -> torch.Tensor:
    if not parameters:
        return torch.tensor(0.0)
    return parameters[0][1].new_zeros(())


class RegularizationStrategy:
    method = "finetune"

    def begin_task(self, parameters: Sequence[tuple[str, nn.Parameter]]) -> None:
        del parameters

    def penalty(self, parameters: Sequence[tuple[str, nn.Parameter]]) -> torch.Tensor:
        return _zero_for(parameters)

    def capture_step(self, parameters: Sequence[tuple[str, nn.Parameter]]) -> None:
        del parameters

    def finish_step(self, parameters: Sequence[tuple[str, nn.Parameter]]) -> None:
        del parameters

    def consolidate(
        self,
        parameters: Sequence[tuple[str, nn.Parameter]],
        importance: dict[str, torch.Tensor] | None = None,
    ) -> None:
        del parameters, importance

    def state_dict(self) -> dict:
        return {"method": self.method}


class QuadraticImportanceStrategy(RegularizationStrategy):
    def __init__(self, method: str, strength: float, decay: float = 1.0):
        if method not in {"ewc", "online_ewc", "mas"}:
            raise ValueError(f"Unsupported quadratic strategy: {method}")
        self.method = method
        self.strength = float(strength)
        self.decay = float(decay)
        self.importance: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}

    def penalty(self, parameters: Sequence[tuple[str, nn.Parameter]]) -> torch.Tensor:
        total = _zero_for(parameters)
        for name, parameter in parameters:
            if name not in self.importance:
                continue
            total = total + (
                self.importance[name]
                * (parameter - self.anchor[name]).pow(2)
            ).sum()
        return self.strength * total

    @torch.no_grad()
    def consolidate(
        self,
        parameters: Sequence[tuple[str, nn.Parameter]],
        importance: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if importance is None:
            raise ValueError(f"{self.method} requires an importance estimate")

        current = _clone_parameters(parameters)
        if not self.importance:
            self.importance = {
                name: importance[name].detach().clone()
                for name in current
            }
            self.anchor = current
            return

        if self.method == "ewc":
            # A sum of task-wise diagonal quadratics can be represented exactly
            # by a cumulative precision and its precision-weighted center.
            for name, value in current.items():
                old_importance = self.importance[name]
                new_task_importance = importance[name].detach()
                combined = old_importance + new_task_importance
                numerator = (
                    old_importance * self.anchor[name]
                    + new_task_importance * value
                )
                self.anchor[name] = torch.where(
                    combined > 0,
                    numerator / combined.clamp_min(1e-30),
                    value,
                )
                self.importance[name] = combined
        else:
            for name, value in current.items():
                self.importance[name] = (
                    self.decay * self.importance[name]
                    + importance[name].detach()
                )
                self.anchor[name] = value

    def state_dict(self) -> dict:
        return {
            "method": self.method,
            "strength": self.strength,
            "decay": self.decay,
            "importance": {
                name: tensor.detach().cpu() for name, tensor in self.importance.items()
            },
            "anchor": {
                name: tensor.detach().cpu() for name, tensor in self.anchor.items()
            },
        }


class SynapticIntelligenceStrategy(RegularizationStrategy):
    method = "si"

    def __init__(self, strength: float, xi: float):
        self.strength = float(strength)
        self.xi = float(xi)
        self.importance: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}
        self.task_start: dict[str, torch.Tensor] = {}
        self.path_integral: dict[str, torch.Tensor] = {}
        self.pre_step: dict[str, torch.Tensor] = {}
        self.step_gradient: dict[str, torch.Tensor] = {}

    def begin_task(self, parameters: Sequence[tuple[str, nn.Parameter]]) -> None:
        self.task_start = _clone_parameters(parameters)
        self.path_integral = {
            name: torch.zeros_like(parameter)
            for name, parameter in parameters
        }
        self.pre_step = {}
        self.step_gradient = {}

    def penalty(self, parameters: Sequence[tuple[str, nn.Parameter]]) -> torch.Tensor:
        total = _zero_for(parameters)
        for name, parameter in parameters:
            if name not in self.importance:
                continue
            total = total + (
                self.importance[name]
                * (parameter - self.anchor[name]).pow(2)
            ).sum()
        return self.strength * total

    def capture_step(self, parameters: Sequence[tuple[str, nn.Parameter]]) -> None:
        self.pre_step = _clone_parameters(parameters)
        self.step_gradient = {
            name: parameter.grad.detach().clone()
            for name, parameter in parameters
            if parameter.grad is not None
        }

    @torch.no_grad()
    def finish_step(self, parameters: Sequence[tuple[str, nn.Parameter]]) -> None:
        for name, parameter in parameters:
            if name not in self.step_gradient:
                continue
            update = parameter.detach() - self.pre_step[name]
            self.path_integral[name].add_(-self.step_gradient[name] * update)
        self.pre_step = {}
        self.step_gradient = {}

    @torch.no_grad()
    def consolidate(
        self,
        parameters: Sequence[tuple[str, nn.Parameter]],
        importance: dict[str, torch.Tensor] | None = None,
    ) -> None:
        del importance
        current = _clone_parameters(parameters)
        for name, value in current.items():
            movement = value - self.task_start[name]
            contribution = self.path_integral[name] / (movement.pow(2) + self.xi)
            contribution = contribution.clamp_min(0.0)
            if name in self.importance:
                self.importance[name].add_(contribution)
            else:
                self.importance[name] = contribution.clone()
            self.anchor[name] = value

    def state_dict(self) -> dict:
        return {
            "method": self.method,
            "strength": self.strength,
            "xi": self.xi,
            "importance": {
                name: tensor.detach().cpu() for name, tensor in self.importance.items()
            },
            "anchor": {
                name: tensor.detach().cpu() for name, tensor in self.anchor.items()
            },
        }


def build_regularization_strategy(
    method: str,
    *,
    ewc_strength: float,
    online_ewc_strength: float,
    online_ewc_decay: float,
    si_strength: float,
    si_xi: float,
    mas_strength: float,
    mas_decay: float,
) -> RegularizationStrategy:
    if method == "finetune":
        return RegularizationStrategy()
    if method == "ewc":
        return QuadraticImportanceStrategy("ewc", ewc_strength)
    if method == "online_ewc":
        return QuadraticImportanceStrategy(
            "online_ewc", online_ewc_strength, online_ewc_decay
        )
    if method == "si":
        return SynapticIntelligenceStrategy(si_strength, si_xi)
    if method == "mas":
        return QuadraticImportanceStrategy("mas", mas_strength, mas_decay)
    raise ValueError(f"Unknown regularization method: {method}")
