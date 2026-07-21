import unittest

import torch

from experiments.generate_frozen_proxy_frequency_shift import (
    balanced_rademacher,
    budget_name,
    nested_attack_tasks,
    project_direction,
    sequence_metrics,
)


class FrozenProxyFrequencyShiftTest(unittest.TestCase):
    def test_nested_task_sets(self):
        frequent, infrequent = nested_attack_tasks(49, 25, 3)
        self.assertEqual(len(frequent), 25)
        self.assertEqual(len(infrequent), 3)
        self.assertTrue(set(infrequent).issubset(frequent))
        self.assertEqual(frequent[0], 1)
        self.assertEqual(frequent[-1], 49)

    def test_balanced_rademacher_is_reproducible_and_centered(self):
        first = balanced_rademacher(10, 123)
        second = balanced_rademacher(10, 123)
        self.assertTrue((first == second).all())
        self.assertEqual(set(first.tolist()), {-1, 1})
        self.assertEqual(int(first.sum()), 0)
        odd = balanced_rademacher(9, 456)
        self.assertLessEqual(abs(int(odd.sum())), 1)

    def test_projection_respects_both_budgets_and_is_symmetric(self):
        generator = torch.Generator().manual_seed(7)
        base = torch.randn(3, 2, 4, 20, generator=generator)
        direction = torch.randn(3, 2, 4, 20, generator=generator)
        projected = project_direction(
            direction,
            base,
            linf_std_scale=0.1,
            relative_l2=0.05,
        )
        negative = project_direction(
            -direction,
            base,
            linf_std_scale=0.1,
            relative_l2=0.05,
        )
        self.assertTrue(torch.allclose(negative, -projected))
        flat_base = base.reshape(base.shape[0], -1)
        flat_delta = projected.reshape(projected.shape[0], -1)
        relative_l2 = torch.linalg.vector_norm(flat_delta, dim=1) / torch.linalg.vector_norm(
            flat_base, dim=1
        )
        linf_over_std = flat_delta.abs().amax(dim=1) / flat_base.std(
            dim=1, unbiased=False
        )
        self.assertTrue((relative_l2 <= 0.05 + 1e-6).all())
        self.assertTrue((linf_over_std <= 0.1 + 1e-6).all())

    def test_budget_directory_name_is_stable(self):
        self.assertEqual(budget_name(0.05), "rel_l2_0500")
        self.assertEqual(budget_name(0.025), "rel_l2_0250")

    def test_signal_diagnostics_are_finite_and_bounded(self):
        time = torch.arange(3000) / 100.0
        base = torch.sin(2 * torch.pi * 10 * time).reshape(1, 1, 1, -1)
        delta = 0.01 * torch.sin(2 * torch.pi * 12 * time).reshape(1, 1, 1, -1)
        metrics = sequence_metrics(
            base,
            delta,
            sample_rate=100.0,
            band_low=0.3,
            band_high=35.0,
        )
        self.assertGreaterEqual(metrics["spectral_total_variation"][0], 0.0)
        self.assertLessEqual(metrics["spectral_total_variation"][0], 1.0)
        self.assertLess(metrics["perturbation_out_of_band_fraction"][0], 1e-5)
        self.assertGreaterEqual(metrics["sample_outside_clean_range_fraction"][0], 0.0)
        self.assertLessEqual(metrics["sample_outside_clean_range_fraction"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
