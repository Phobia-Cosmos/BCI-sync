from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.progressive_feedback_proxy import (
    ProgressiveFeedbackProxy,
    _constrain_step_to_direction,
    _eeg_invariant_descriptor,
    _fill_step_relative_l2,
    _fit_eeg_invariants,
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
            progressive_upload_full_pool=True,
            progressive_match_task_sequence_count=False,
            progressive_active_fraction=1.0,
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_progressive_proxy_args(args, 4)

    def test_feedback_requires_full_pool_and_all_sequences_active(self):
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
            progressive_clean_feedback_tasks="2,4",
            progressive_upload_full_pool=False,
            progressive_match_task_sequence_count=False,
            progressive_active_fraction=1.0,
        )
        with self.assertRaisesRegex(ValueError, "full-pool"):
            validate_progressive_proxy_args(args, 4)
        args.progressive_upload_full_pool = True
        args.progressive_active_fraction = 0.5
        with self.assertRaisesRegex(ValueError, "active-fraction 1"):
            validate_progressive_proxy_args(args, 4)

    def test_feedback_accepts_task_matched_cardinality(self):
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
            progressive_proxy_tasks="1,2",
            progressive_clean_feedback_tasks="3,4",
            progressive_upload_full_pool=False,
            progressive_match_task_sequence_count=True,
            progressive_active_fraction=1.0,
        )
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

    def test_proxy_pool_expands_for_larger_natural_task(self):
        proxy = ProgressiveFeedbackProxy.__new__(ProgressiveFeedbackProxy)
        proxy.initial_pool = np.arange(2 * 2 * 1 * 4, dtype=np.float32).reshape(
            2, 2, 1, 4
        )
        proxy.current_pool = proxy.initial_pool + 1.0
        proxy.active_mask = np.array([True, True])
        proxy.input_direction = proxy.current_pool - proxy.initial_pool

        proxy._ensure_pool_capacity(5)

        self.assertEqual(len(proxy.current_pool), 5)
        self.assertEqual(proxy.active_mask.tolist(), [True] * 5)
        np.testing.assert_array_equal(proxy.initial_pool[2], proxy.initial_pool[0])
        np.testing.assert_array_equal(proxy.current_pool[4], proxy.current_pool[0])
        np.testing.assert_array_equal(proxy.input_direction[3], proxy.input_direction[1])

    def test_proxy_diagnostic_labels_cycle_when_task_has_more_sequences(self):
        proxy = ProgressiveFeedbackProxy.__new__(ProgressiveFeedbackProxy)
        proxy.proxy_tasks = {1}
        proxy.base_label_paths = [Path("0.npy"), Path("1.npy")]
        proxy.args = SimpleNamespace(progressive_match_task_sequence_count=True)

        labels = proxy.diagnostic_label_paths(
            1,
            [Path(f"{index}.npy") for index in range(5)],
        )

        self.assertEqual(labels, [Path("0.npy"), Path("1.npy"), Path("0.npy"), Path("1.npy"), Path("0.npy")])

    def test_population_pool_is_class_balanced_and_cross_subject(self):
        proxy = ProgressiveFeedbackProxy.__new__(ProgressiveFeedbackProxy)
        proxy.args = SimpleNamespace(
            model_param=SimpleNamespace(NumClasses=3),
            progressive_feedback_weight=1.0,
        )
        proxy.rng = np.random.default_rng(123)
        proxy.feedback_records = []
        proxy.labeled_records = []
        proxy.population_class_counts = []
        proxy.population_subject_count = 0
        for subject, class_index in enumerate((0, 1, 2, 0, 1, 2)):
            data = np.full((2, 8, 64), float(subject + 1), dtype=np.float32)
            probabilities = np.zeros((2, 3), dtype=np.float32)
            probabilities[:, class_index] = 1.0
            proxy.feedback_records.append({
                "data": data,
                "probabilities": probabilities,
                "subject": subject,
                "kind": "proxy",
            })
        pool = proxy._build_population_pool(6)
        self.assertEqual(pool.shape, (6, 2, 8, 64))
        self.assertEqual(pool.dtype, np.float32)
        self.assertEqual(proxy.population_class_counts, [2, 2, 2])
        self.assertGreaterEqual(proxy.population_subject_count, 3)
        self.assertEqual(proxy.population_record_kind_counts["proxy"], 6)

    def test_invariant_fit_is_finite_nontrivial_eeg_shape(self):
        rng = np.random.default_rng(321)
        first = rng.normal(size=(20, 8, 128)).astype(np.float32)
        second = rng.normal(size=(20, 8, 128)).astype(np.float32)
        fitted = _fit_eeg_invariants(first, second, 0.55)
        self.assertEqual(fitted.shape, first.shape)
        self.assertTrue(np.isfinite(fitted).all())
        self.assertFalse(np.allclose(fitted, first))
        descriptor = _eeg_invariant_descriptor(fitted)
        self.assertEqual(descriptor.shape, (15,))
        self.assertTrue(np.isfinite(descriptor).all())

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
