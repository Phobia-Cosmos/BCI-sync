import inspect
import unittest

import torch

from model.icml2026_cl_defenses import (
    RobustFeatureDefense,
    T2TDetector,
    diagonal_t2t_score,
    robust_feature_eigenvalues,
)
from experiments.regularization_cl_eeg import (
    estimate_classifier_feature_covariance,
)


class ICML2026DefenseTest(unittest.TestCase):
    def test_defense_interfaces_do_not_accept_target_labels(self):
        for function in (
            diagonal_t2t_score,
            robust_feature_eigenvalues,
            estimate_classifier_feature_covariance,
        ):
            names = set(inspect.signature(function).parameters)
            self.assertNotIn("labels", names)
            self.assertNotIn("true_labels", names)

    def test_diagonal_t2t_matches_dense_moore_penrose_equation(self):
        q_t = torch.tensor([2.0, 3.0, 4.0])
        q_tm1 = torch.tensor([1.5, 2.5, 3.5])
        h_t = torch.tensor([0.5, 0.75, 1.0])
        h_tm1 = torch.tensor([0.25, 0.5, 0.8])
        w_tm2 = torch.tensor([0.1, -0.3, 0.4])
        w_tm1 = torch.tensor([0.2, -0.1, 0.35])
        w_t = torch.tensor([0.25, -0.15, 0.5])

        result = diagonal_t2t_score(
            {"block.weight": w_t},
            {"block.weight": w_tm1},
            {"block.weight": w_tm2},
            {"block.weight": q_t},
            {"block.weight": q_tm1},
            {"block.weight": h_t},
            {"block.weight": h_tm1},
            pinv_rtol=0.0,
            eps=1e-15,
        )

        q_t_dense = torch.diag(q_t)
        q_tm1_dense = torch.diag(q_tm1)
        h_t_dense = torch.diag(h_t)
        h_tm1_dense = torch.diag(h_tm1)
        a = (
            torch.linalg.inv(q_t_dense + h_t_dense)
            @ q_t_dense
            @ torch.linalg.inv(q_tm1_dense + h_tm1_dense)
            @ h_tm1_dense
        )
        b = torch.linalg.inv(q_tm1_dense + h_tm1_dense) @ q_tm1_dense
        c = (
            torch.linalg.pinv(a) @ a.T - torch.linalg.pinv(b) @ b.T
        ) @ torch.linalg.pinv(a @ a.T + b @ b.T)
        identity = torch.eye(3)
        d1 = (identity - b) @ (torch.linalg.pinv(a) - c @ a.T)
        d2 = (identity - b) @ (torch.linalg.pinv(b) + c @ b.T)
        expected = torch.linalg.vector_norm(
            d1 @ (w_t - w_tm1) - d2 @ (w_tm1 - w_tm2)
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["active_dimensions"], 3)
        self.assertAlmostEqual(result["score"], float(expected), places=6)

    def test_diagonal_t2t_removes_non_common_subspace(self):
        result = diagonal_t2t_score(
            {"p": torch.tensor([1.0, 2.0])},
            {"p": torch.tensor([0.0, 1.0])},
            {"p": torch.tensor([-1.0, 0.0])},
            {"p": torch.tensor([1.0, 1.0])},
            {"p": torch.tensor([1.0, 0.0])},
            {"p": torch.tensor([1.0, 1.0])},
            {"p": torch.tensor([0.0, 1.0])},
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["score"], 0.0)

    def test_t2t_detector_uses_previous_window_only(self):
        detector = T2TDetector(
            threshold_multiplier=2.5,
            window=2,
            minimum_history=1,
        )
        first = detector.decide({"valid": True, "score": 2.0})
        second = detector.decide({"valid": True, "score": 4.9})
        third = detector.decide({"valid": True, "score": 10.0})
        self.assertFalse(first["detected"])
        self.assertAlmostEqual(second["threshold"], 5.0)
        self.assertFalse(second["detected"])
        self.assertAlmostEqual(third["moving_mean"], 3.45)
        self.assertTrue(third["detected"])

    def test_robust_feature_solution_satisfies_kkt_sensitivity(self):
        gamma = torch.tensor([[0.2, 0.7, 1.5, 4.0]])
        risk = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        sample_count = 20
        sigma2 = 1.0
        budget = 8.0
        regularizer, allocation, diagnostics = robust_feature_eigenvalues(
            gamma,
            risk,
            sample_count=sample_count,
            sigma2=sigma2,
            budget=budget,
        )

        self.assertGreater(diagnostics["protected_dimensions"], 0)
        self.assertAlmostEqual(float(allocation.sum()), budget, places=4)
        contraction = regularizer / (gamma + regularizer)
        sensitivity = (1.0 - contraction).square() / (gamma * sample_count)
        protected = allocation > 1e-6
        protected_values = sensitivity[protected]
        self.assertLess(
            float(protected_values.max() - protected_values.min()),
            1e-5,
        )
        if (~protected).any():
            self.assertLessEqual(
                float(sensitivity[~protected].max()),
                float(protected_values.max()) + 1e-6,
            )

    def test_robust_feature_penalty_is_zero_at_anchor(self):
        weight = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
        covariance = torch.tensor([[2.0, 0.3], [0.3, 0.5]])
        defense = RobustFeatureDefense(
            sigma2=1.0,
            budget_per_dimension=0.1,
        )
        diagnostics = defense.prepare_task(weight, covariance, sample_count=10)
        self.assertAlmostEqual(float(defense.penalty(weight)), 0.0, places=7)
        moved = weight.clone()
        moved[0, 0] += 0.2
        self.assertGreater(float(defense.penalty(moved)), 0.0)
        finished = defense.finish_task()
        self.assertIn("risk_mean_after", finished)
        self.assertEqual(diagnostics["dimensions"], weight.numel())


if __name__ == "__main__":
    unittest.main()
