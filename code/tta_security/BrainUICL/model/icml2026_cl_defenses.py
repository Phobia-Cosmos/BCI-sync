from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


TensorMap = Mapping[str, torch.Tensor]


def _safe_denominator(value: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.where(value.abs() > eps, value, torch.full_like(value, eps))


def diagonal_t2t_score(
    current_parameters: TensorMap,
    previous_parameters: TensorMap,
    previous_previous_parameters: TensorMap,
    current_hessian: TensorMap,
    previous_hessian: TensorMap,
    current_regularizer: TensorMap,
    previous_regularizer: TensorMap,
    *,
    pinv_rtol: float = 1e-6,
    eps: float = 1e-12,
) -> dict[str, float | int | bool | dict[str, float | int]]:
    """Compute Eq. (6) using the paper's diagonal-Hessian approximation.

    For diagonal A and B, the coupling matrix C is zero on their common
    support and both projected coefficients are zero outside that support.
    This lets us evaluate the Moore-Penrose expression elementwise without
    materializing p-by-p matrices.
    """

    names = tuple(current_parameters)
    collections = (
        previous_parameters,
        previous_previous_parameters,
        current_hessian,
        previous_hessian,
        current_regularizer,
        previous_regularizer,
    )
    for values in collections:
        if set(values) != set(names):
            raise ValueError("All T2T tensor maps must contain the same parameters")

    score_squared = 0.0
    current_update_squared = 0.0
    previous_update_squared = 0.0
    active_dimensions = 0
    total_dimensions = 0
    blocks: dict[str, dict[str, float | int]] = {}

    for name in names:
        w_t = current_parameters[name]
        w_tm1 = previous_parameters[name].to(device=w_t.device, dtype=w_t.dtype)
        w_tm2 = previous_previous_parameters[name].to(device=w_t.device, dtype=w_t.dtype)
        q_t = current_hessian[name].to(device=w_t.device, dtype=w_t.dtype).clamp_min(0)
        q_tm1 = previous_hessian[name].to(device=w_t.device, dtype=w_t.dtype).clamp_min(0)
        h_t = current_regularizer[name].to(device=w_t.device, dtype=w_t.dtype).clamp_min(0)
        h_tm1 = previous_regularizer[name].to(device=w_t.device, dtype=w_t.dtype).clamp_min(0)

        if not (
            w_t.shape
            == w_tm1.shape
            == w_tm2.shape
            == q_t.shape
            == q_tm1.shape
            == h_t.shape
            == h_tm1.shape
        ):
            raise ValueError(f"T2T shape mismatch for {name}")

        s_t = _safe_denominator(q_t + h_t, eps)
        s_tm1 = _safe_denominator(q_tm1 + h_tm1, eps)
        a_t = (q_t / s_t) * (h_tm1 / s_tm1)
        b_t = q_tm1 / s_tm1

        a_scale = float(a_t.detach().abs().max().cpu()) if a_t.numel() else 0.0
        b_scale = float(b_t.detach().abs().max().cpu()) if b_t.numel() else 0.0
        a_tol = max(eps, pinv_rtol * a_scale)
        b_tol = max(eps, pinv_rtol * b_scale)
        support = (a_t.abs() > a_tol) & (b_t.abs() > b_tol)

        delta_t = w_t - w_tm1
        delta_tm1 = w_tm1 - w_tm2
        projected = torch.zeros_like(delta_t)
        if support.any():
            one_minus_b = 1.0 - b_t[support]
            d_t1 = one_minus_b / a_t[support]
            d_t2 = one_minus_b / b_t[support]
            projected[support] = (
                d_t1 * delta_t[support] - d_t2 * delta_tm1[support]
            )

        block_name = name.split(".", 1)[0]
        block = blocks.setdefault(
            block_name,
            {"score_squared": 0.0, "active_dimensions": 0, "total_dimensions": 0},
        )
        local_score_squared = float(projected.double().square().sum().cpu())
        local_active = int(support.sum().item())
        local_total = int(support.numel())
        block["score_squared"] = float(block["score_squared"]) + local_score_squared
        block["active_dimensions"] = int(block["active_dimensions"]) + local_active
        block["total_dimensions"] = int(block["total_dimensions"]) + local_total

        score_squared += local_score_squared
        current_update_squared += float(delta_t.double().square().sum().cpu())
        previous_update_squared += float(delta_tm1.double().square().sum().cpu())
        active_dimensions += local_active
        total_dimensions += local_total

    block_summary = {
        name: {
            "score": float(float(values["score_squared"]) ** 0.5),
            "active_dimensions": int(values["active_dimensions"]),
            "total_dimensions": int(values["total_dimensions"]),
        }
        for name, values in blocks.items()
    }
    score = score_squared**0.5
    return {
        "valid": active_dimensions > 0,
        "score": float(score),
        "score_rms": float((score_squared / max(active_dimensions, 1)) ** 0.5),
        "active_dimensions": active_dimensions,
        "total_dimensions": total_dimensions,
        "active_fraction": active_dimensions / max(total_dimensions, 1),
        "current_update_norm": float(current_update_squared**0.5),
        "previous_update_norm": float(previous_update_squared**0.5),
        "blocks": block_summary,
    }


@dataclass
class T2TDetector:
    threshold_multiplier: float = 2.5
    window: int = 5
    minimum_history: int = 1
    score_floor: float = 1e-12
    scores: list[float] = field(default_factory=list)

    def decide(self, score_diagnostics: dict) -> dict:
        valid = bool(score_diagnostics["valid"])
        previous = self.scores[-self.window :]
        enough_history = len(previous) >= self.minimum_history
        moving_mean = (
            float(sum(previous) / len(previous)) if previous else None
        )
        threshold = (
            self.threshold_multiplier * max(moving_mean, self.score_floor)
            if valid and enough_history and moving_mean is not None
            else None
        )
        detected = bool(
            valid
            and threshold is not None
            and float(score_diagnostics["score"]) >= threshold
        )
        if valid:
            self.scores.append(float(score_diagnostics["score"]))
        return {
            **score_diagnostics,
            "moving_mean": moving_mean,
            "threshold": threshold,
            "threshold_multiplier": self.threshold_multiplier,
            "window": self.window,
            "history_count": len(previous),
            "detected": detected,
        }

    def state_dict(self) -> dict:
        return {
            "threshold_multiplier": self.threshold_multiplier,
            "window": self.window,
            "minimum_history": self.minimum_history,
            "score_floor": self.score_floor,
            "scores": list(self.scores),
        }


def robust_feature_eigenvalues(
    gamma: torch.Tensor,
    previous_risk: torch.Tensor,
    *,
    sample_count: int,
    sigma2: float,
    budget: float,
    eps: float = 1e-12,
    max_regularizer: float = 1e6,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Solve Proposition 5.2 in a supplied common eigenbasis.

    ``gamma`` and ``previous_risk`` may be matrices. They are flattened only
    for the active-set calculation and are returned in their original shape.
    """

    if gamma.shape != previous_risk.shape:
        raise ValueError("gamma and previous_risk must have the same shape")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    shape = gamma.shape
    gamma_flat = gamma.reshape(-1).clamp_min(0)
    risk_flat = previous_risk.reshape(-1).clamp_min(0)
    valid = (gamma_flat > eps) & (risk_flat > eps)
    regularizer = torch.full_like(gamma_flat, max_regularizer)
    adversarial_allocation = torch.zeros_like(gamma_flat)

    positive_risk = risk_flat > eps
    regularizer[positive_risk] = (
        sigma2 / (sample_count * risk_flat[positive_risk])
    ).clamp(max=max_regularizer)

    protected_count = 0
    equilibrium_level = None
    if valid.any():
        valid_indices = torch.where(valid)[0]
        gamma_tilde = gamma_flat[valid] * sample_count
        risk = risk_flat[valid]
        sqrt_gamma_tilde = gamma_tilde.sqrt()
        b = (risk * gamma_tilde + sigma2) / (
            risk * sqrt_gamma_tilde
        ).clamp_min(eps)
        order = torch.argsort(b)
        sorted_indices = valid_indices[order]
        sorted_gamma_tilde = gamma_tilde[order]
        sorted_risk = risk[order]
        sorted_b = b[order]
        positions = torch.arange(
            1,
            len(sorted_indices) + 1,
            device=gamma.device,
            dtype=gamma.dtype,
        )
        cumulative_numerator = torch.cumsum(
            sorted_risk * sorted_gamma_tilde,
            dim=0,
        )
        cumulative_denominator = torch.cumsum(
            sorted_risk * sorted_gamma_tilde.sqrt(),
            dim=0,
        ).clamp_min(eps)
        a = (
            budget + positions * sigma2 + cumulative_numerator
        ) / cumulative_denominator
        active_candidates = a > sorted_b
        if active_candidates.any():
            protected_count = int(torch.where(active_candidates)[0].max().item()) + 1
            equilibrium = a[protected_count - 1]
            equilibrium_level = float(equilibrium.detach().cpu())
            protected_indices = sorted_indices[:protected_count]
            protected_gamma = gamma_flat[protected_indices]
            protected_risk = risk_flat[protected_indices]
            protected_gamma_tilde = protected_gamma * sample_count
            protected_lambda = (
                equilibrium * (protected_gamma / sample_count).sqrt()
                - protected_gamma
            ).clamp(min=eps, max=max_regularizer)
            protected_chi = (
                protected_risk * protected_gamma_tilde.sqrt() * equilibrium
                - sigma2
                - protected_risk * protected_gamma_tilde
            ).clamp_min(0)
            regularizer[protected_indices] = protected_lambda
            adversarial_allocation[protected_indices] = protected_chi

    regularizer = regularizer.reshape(shape)
    adversarial_allocation = adversarial_allocation.reshape(shape)
    diagnostics = {
        "dimensions": int(gamma_flat.numel()),
        "informative_dimensions": int(valid.sum().item()),
        "protected_dimensions": protected_count,
        "protected_fraction": protected_count / max(int(gamma_flat.numel()), 1),
        "budget": float(budget),
        "sigma2": float(sigma2),
        "equilibrium_level": equilibrium_level,
        "lambda_min": float(regularizer.min().detach().cpu()),
        "lambda_mean": float(regularizer.mean().detach().cpu()),
        "lambda_max": float(regularizer.max().detach().cpu()),
        "allocated_budget": float(adversarial_allocation.sum().detach().cpu()),
    }
    return regularizer, adversarial_allocation, diagnostics


class RobustFeatureDefense:
    """Eq. (14)-(15) defense on the final linear classifier feature space."""

    def __init__(
        self,
        *,
        sigma2: float,
        budget_per_dimension: float,
        initial_risk: float = 0.0,
        eps: float = 1e-12,
        max_regularizer: float = 1e6,
    ):
        self.sigma2 = float(sigma2)
        self.budget_per_dimension = float(budget_per_dimension)
        self.initial_risk = float(initial_risk)
        self.eps = float(eps)
        self.max_regularizer = float(max_regularizer)
        self.risk_covariance: torch.Tensor | None = None
        self.anchor: torch.Tensor | None = None
        self.basis: torch.Tensor | None = None
        self.gamma: torch.Tensor | None = None
        self.previous_risk: torch.Tensor | None = None
        self.regularizer: torch.Tensor | None = None
        self.adversarial_allocation: torch.Tensor | None = None
        self.sample_count = 0
        self.last_diagnostics: dict = {}

    def prepare_task(
        self,
        classifier_weight: torch.Tensor,
        feature_covariance: torch.Tensor,
        sample_count: int,
    ) -> dict:
        classes, feature_dim = classifier_weight.shape
        if feature_covariance.shape != (feature_dim, feature_dim):
            raise ValueError("Feature covariance does not match classifier input")

        covariance = 0.5 * (feature_covariance + feature_covariance.T)
        gamma, basis = torch.linalg.eigh(covariance)
        gamma = gamma.clamp_min(self.eps)
        if self.risk_covariance is None:
            risk = self.initial_risk
            if risk <= 0:
                risk = float(classifier_weight.detach().square().mean().cpu())
            risk = max(risk, self.eps)
            identity = torch.eye(
                feature_dim,
                device=classifier_weight.device,
                dtype=classifier_weight.dtype,
            )
            self.risk_covariance = (
                identity.unsqueeze(0).repeat(classes, 1, 1) * risk
            )
        if self.risk_covariance.shape != (classes, feature_dim, feature_dim):
            raise ValueError("Stored robust-feature risk has an incompatible shape")

        risk_in_basis = torch.stack(
            [
                torch.diagonal(basis.T @ covariance_row @ basis)
                for covariance_row in self.risk_covariance
            ]
        ).clamp_min(self.eps)
        gamma_by_output = gamma.unsqueeze(0).expand(classes, -1)
        budget = self.budget_per_dimension * classes * feature_dim
        regularizer, allocation, diagnostics = robust_feature_eigenvalues(
            gamma_by_output,
            risk_in_basis,
            sample_count=sample_count,
            sigma2=self.sigma2,
            budget=budget,
            eps=self.eps,
            max_regularizer=self.max_regularizer,
        )

        self.anchor = classifier_weight.detach().clone()
        self.basis = basis
        self.gamma = gamma_by_output
        self.previous_risk = risk_in_basis
        self.regularizer = regularizer
        self.adversarial_allocation = allocation
        self.sample_count = int(sample_count)
        diagnostics.update(
            {
                "feature_dimension": feature_dim,
                "classes": classes,
                "sample_count": sample_count,
                "feature_eigenvalue_min": float(gamma.min().detach().cpu()),
                "feature_eigenvalue_median": float(gamma.median().detach().cpu()),
                "feature_eigenvalue_max": float(gamma.max().detach().cpu()),
                "risk_mean_before": float(risk_in_basis.mean().detach().cpu()),
            }
        )
        self.last_diagnostics = diagnostics
        return dict(diagnostics)

    def penalty(self, classifier_weight: torch.Tensor) -> torch.Tensor:
        if self.anchor is None or self.basis is None or self.regularizer is None:
            return classifier_weight.new_zeros(())
        delta_in_basis = (classifier_weight - self.anchor) @ self.basis
        return 0.5 * (self.regularizer * delta_in_basis.square()).sum()

    @torch.no_grad()
    def finish_task(self) -> dict:
        required = (
            self.basis,
            self.gamma,
            self.previous_risk,
            self.regularizer,
            self.adversarial_allocation,
        )
        if any(value is None for value in required):
            raise RuntimeError("prepare_task must be called before finish_task")

        gamma = self.gamma
        risk = self.previous_risk
        regularizer = self.regularizer
        allocation = self.adversarial_allocation
        informative = gamma > self.eps
        contraction = regularizer / (gamma + regularizer).clamp_min(self.eps)
        innovation = torch.zeros_like(risk)
        innovation[informative] = (
            self.sigma2 + allocation[informative]
        ) / (self.sample_count * gamma[informative]).clamp_min(self.eps)
        updated_risk = contraction.square() * risk + (
            1.0 - contraction
        ).square() * innovation
        updated_risk = torch.where(informative, updated_risk, risk).clamp_min(self.eps)
        self.risk_covariance = torch.stack(
            [
                self.basis @ torch.diag(row) @ self.basis.T
                for row in updated_risk
            ]
        )
        self.last_diagnostics = {
            **self.last_diagnostics,
            "risk_mean_after": float(updated_risk.mean().detach().cpu()),
            "risk_max_after": float(updated_risk.max().detach().cpu()),
        }
        return dict(self.last_diagnostics)

    def state_dict(self) -> dict:
        return {
            "sigma2": self.sigma2,
            "budget_per_dimension": self.budget_per_dimension,
            "initial_risk": self.initial_risk,
            "eps": self.eps,
            "max_regularizer": self.max_regularizer,
            "risk_covariance": (
                self.risk_covariance.detach().cpu()
                if self.risk_covariance is not None
                else None
            ),
            "last_diagnostics": dict(self.last_diagnostics),
        }
