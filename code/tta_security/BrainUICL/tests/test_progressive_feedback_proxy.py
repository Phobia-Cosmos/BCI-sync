from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import numpy as np

from experiments.progressive_feedback_proxy import (
    ProgressiveFeedbackProxy,
    _constrain_step_to_direction,
    _fill_step_relative_l2,
    _project_numpy,
    resolve_task_spec,
    validate_progressive_proxy_args,
)


class ProgressiveFeedbackProxyTests(unittest.TestCase):
    def test_odd_even_schedule_preserves_all_tasks_without_overlap(self):
        odd = resolve_task_spec("odd", 49)
        even = resolve_task_spec("even", 49)
        self.assertEqual(len(odd), 25)
        self.assertEqual(len(even), 24)
        self.assertFalse(odd & even)
        self.assertEqual(odd | even, set(range(1, 50)))

    def test_schedule_validation_rejects_overlapping_visible_roles(self):
        args = SimpleNamespace(
            progressive_proxy_mode="feedback",
            progressive_proxy_lr=1e-6,
            progressive_feedback_batch=2,
            progressive_generation_steps=1,
            progressive_generation_batch=2,
            progressive_reference_batch=1,
            progressive_step_relative_l2=0.01,
            progressive_step_linf_std=0.02,
            progressive_cumulative_relative_l2=0.1,
            progressive_cumulative_linf_std=0.2,
            progressive_feedback_capacity=10,
            progressive_feedback_steps=1,
            progressive_guide_epochs=1,
            progressive_history_decay=0.8,
            progressive_feedback_decay=0.95,
            progressive_input_cone_residual=0.5,
            progressive_proxy_tasks="1,3",
            progressive_clean_feedback_tasks="2,3",
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_progressive_proxy_args(args, 4)

    def test_numpy_projection_enforces_relative_and_linf_budgets(self):
        rng = np.random.default_rng(7)
        base = rng.normal(size=(3, 2, 4, 32)).astype(np.float32)
        candidate = base + rng.normal(size=base.shape).astype(np.float32) * 10
        projected = _project_numpy(
            candidate,
            base,
            max_relative_l2=0.1,
            max_linf_over_std=0.2,
        )
        delta = projected - base
        for channel_slice in (slice(None), slice(0, 2), slice(2, None)):
            local_base = base[:, :, channel_slice, :]
            local_delta = delta[:, :, channel_slice, :]
            relative = np.linalg.norm(local_delta.reshape(3, -1), axis=1) / np.linalg.norm(
                local_base.reshape(3, -1), axis=1
            )
            linf_over_std = np.max(np.abs(local_delta), axis=(1, 2, 3)) / np.std(
                local_base, axis=(1, 2, 3)
            )
            self.assertTrue(np.all(relative <= 0.1 + 1e-6))
            self.assertTrue(np.all(linf_over_std <= 0.2 + 1e-6))

    def test_cone_constraint_keeps_positive_historical_projection(self):
        direction = np.ones((2, 2, 3, 8), dtype=np.float32)
        step = -direction.copy()
        constrained, mean_cosine = _constrain_step_to_direction(
            step,
            direction,
            residual_ratio=0.5,
        )
        dots = (constrained.reshape(2, -1) * direction.reshape(2, -1)).sum(axis=1)
        self.assertTrue(np.all(dots > 0))
        self.assertGreater(mean_cosine, 0)

    def test_fill_step_budget_preserves_direction_and_sets_length(self):
        rng = np.random.default_rng(17)
        base = rng.normal(size=(3, 2, 4, 32)).astype(np.float32)
        step = rng.normal(size=base.shape).astype(np.float32)
        filled = _fill_step_relative_l2(step, base, 0.05)
        relative = np.linalg.norm(filled.reshape(3, -1), axis=1) / np.linalg.norm(
            base.reshape(3, -1), axis=1
        )
        cosine = (filled.reshape(3, -1) * step.reshape(3, -1)).sum(axis=1) / (
            np.linalg.norm(filled.reshape(3, -1), axis=1)
            * np.linalg.norm(step.reshape(3, -1), axis=1)
        )
        np.testing.assert_allclose(relative, 0.05, atol=1e-6)
        np.testing.assert_allclose(cosine, 1.0, atol=1e-6)

    def test_controller_interface_receives_scores_not_victim_blocks(self):
        prepare = set(inspect.signature(ProgressiveFeedbackProxy.prepare_task).parameters)
        observe = set(inspect.signature(ProgressiveFeedbackProxy.observe_task).parameters)
        for forbidden in ("student_blocks", "victim_blocks", "victim_parameters"):
            self.assertNotIn(forbidden, prepare)
            self.assertNotIn(forbidden, observe)
        self.assertIn("victim_probabilities", observe)
        self.assertNotIn("clean_label_paths", observe)


if __name__ == "__main__":
    unittest.main()
