from __future__ import annotations

import unittest

import numpy as np

from experiments.generate_ewc_attack_strength_sweep import (
    STRENGTH_LEVELS,
    build_conditions,
    nested_subset,
    project_modality,
)


class EwcAttackStrengthSweepTests(unittest.TestCase):
    def test_condition_matrix_is_unique_and_complete(self):
        conditions = build_conditions()
        names = [row["name"] for row in conditions]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len([name for name in names if name.startswith("strength_")]), 5)
        self.assertEqual(len([name for name in names if name.startswith("subjects_")]), 4)
        self.assertEqual(len([name for name in names if name.startswith("sequences_")]), 3)
        self.assertEqual(len(conditions), 12)

    def test_nested_subset_preserves_endpoints(self):
        values = tuple(range(1, 50, 2))
        self.assertEqual(nested_subset(values, 1), [1])
        self.assertEqual(nested_subset(values, 3), [1, 25, 49])
        self.assertEqual(nested_subset(values, 25), list(values))

    def test_projection_respects_both_modality_budgets(self):
        rng = np.random.default_rng(123)
        base = rng.normal(size=(2, 20, 300)).astype(np.float32)
        raw = rng.normal(size=base.shape).astype(np.float32)
        delta, metrics = project_modality(raw, base, 0.05, 0.20)
        base_norm = np.linalg.norm(base.reshape(-1))
        base_std = base.std()
        self.assertLessEqual(np.linalg.norm(delta.reshape(-1)) / base_norm, 0.05 + 1e-6)
        self.assertLessEqual(np.max(np.abs(delta)) / base_std, 0.20 + 1e-6)
        self.assertLessEqual(metrics["relative_l2"], 0.05 + 1e-6)
        self.assertLessEqual(metrics["linf_over_std"], 0.20 + 1e-6)

    def test_strength_levels_are_monotonic(self):
        budgets = [row[1] for row in STRENGTH_LEVELS]
        linf = [row[2] for row in STRENGTH_LEVELS]
        self.assertEqual(budgets, sorted(budgets))
        self.assertEqual(linf, sorted(linf))


if __name__ == "__main__":
    unittest.main()
