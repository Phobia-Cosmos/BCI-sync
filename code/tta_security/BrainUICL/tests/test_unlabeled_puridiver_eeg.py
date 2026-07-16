import inspect
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from pure_puridiver_eeg import (  # noqa: E402
    CompactEEGClassifier,
    EpochPool,
    PuriDivERMemory,
    puridiver_split,
    reference_subject_split,
)
from unlabeled_puridiver_eeg import (  # noqa: E402
    adapt_guiding_model_cpc,
    assign_all_guiding_pseudo_labels,
    brainuicl_plasticity_summary,
    brainuicl_stability_summary,
    inject_symmetric_pseudo_label_noise,
    initialize_student,
    load_unlabeled_epoch_pool,
    train_or_load_guiding_model,
)


class FixedGuide(nn.Module):
    def forward(self, x, return_features=False):
        logits = torch.zeros((x.shape[0], 5), device=x.device)
        logits[:, 2] = 3.0
        features = torch.ones((x.shape[0], 4), device=x.device)
        return (logits, features) if return_features else logits


class Args:
    device = torch.device("cpu")
    infer_batch_size = 3


class InitializationArgs:
    device = torch.device("cpu")
    seed = 4321
    student_initialization = "guide_copy"


class CpcArgs:
    device = torch.device("cpu")
    seed = 4321
    num_worker = 0
    guide_policy = "cpc_dynamic"
    guide_cpc_sequence_batch = 2
    guide_cpc_prediction_steps = 2
    guide_cpc_learning_rate = 1e-4
    guide_cpc_epochs = 1
    guide_cpc_temperature = 0.1


class UnlabeledPuriDivEREEGTest(unittest.TestCase):
    def make_pool(self, true_labels):
        true_labels = torch.as_tensor(true_labels, dtype=torch.long)
        return EpochPool(
            x=torch.zeros((len(true_labels), 8, 3000), dtype=torch.float16),
            observed_y=torch.full_like(true_labels, -1),
            true_y=true_labels,
            subject_y=torch.full_like(true_labels, 64),
        )

    def test_pseudo_label_function_has_no_confidence_gate(self):
        parameters = inspect.signature(assign_all_guiding_pseudo_labels).parameters
        self.assertNotIn("confidence_threshold", parameters)

    def test_unlabeled_pool_loader_does_not_require_label_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "0.npy"
            np.save(
                data_path,
                np.random.default_rng(0).normal(size=(20, 8, 3000)).astype(np.float32),
            )

            pool, diagnostics = load_unlabeled_epoch_pool(64, [data_path])

        self.assertEqual(len(pool), 20)
        self.assertTrue(torch.all(pool.true_y == -1))
        self.assertTrue(torch.all(pool.observed_y == -1))
        self.assertFalse(diagnostics["annotation_loaded_for_diagnostics"])

    def test_unlabeled_pseudo_diagnostics_are_none_without_annotations(self):
        pool = self.make_pool([-1, -1, -1, -1])

        _, diagnostics = assign_all_guiding_pseudo_labels(FixedGuide(), pool, Args())

        self.assertIsNone(diagnostics["pseudo_label_acc_diagnostic_only"])
        self.assertIsNone(diagnostics["pseudo_label_mf1_diagnostic_only"])

    def test_every_epoch_receives_guiding_argmax_label(self):
        pool = self.make_pool([0, 1, 2, 3, 4, 0, 1])
        pseudo_pool, diagnostics = assign_all_guiding_pseudo_labels(
            FixedGuide(), pool, Args()
        )

        self.assertEqual(len(pseudo_pool), len(pool))
        self.assertTrue(torch.all(pseudo_pool.observed_y == 2))
        self.assertEqual(diagnostics["accepted_epochs"], len(pool))
        self.assertEqual(diagnostics["acceptance_rate"], 1.0)
        self.assertIsNone(diagnostics["confidence_threshold"])

    def test_true_annotations_do_not_change_guiding_pseudo_labels(self):
        first, _ = assign_all_guiding_pseudo_labels(
            FixedGuide(), self.make_pool([0, 0, 0, 0]), Args()
        )
        second, _ = assign_all_guiding_pseudo_labels(
            FixedGuide(), self.make_pool([1, 2, 3, 4]), Args()
        )

        torch.testing.assert_close(first.observed_y, second.observed_y)

    def test_extra_pseudo_noise_is_reproducible_and_never_keeps_selected_label(self):
        pool = self.make_pool([0, 1, 2, 3, 4] * 4)
        pool.observed_y = torch.arange(5).repeat(4)

        first, first_diagnostics = inject_symmetric_pseudo_label_noise(
            pool, noise_rate=0.5, seed=123
        )
        second, second_diagnostics = inject_symmetric_pseudo_label_noise(
            pool, noise_rate=0.5, seed=123
        )
        changed = first.observed_y.ne(pool.observed_y)

        torch.testing.assert_close(first.observed_y, second.observed_y)
        self.assertEqual(int(changed.sum()), first_diagnostics["extra_pseudo_noise_count"])
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertTrue(torch.all(first.observed_y[changed] != pool.observed_y[changed]))

    def test_hidden_annotations_do_not_choose_extra_pseudo_noise(self):
        first_pool = self.make_pool([0, 0, 0, 0, 0, 0])
        second_pool = self.make_pool([1, 2, 3, 4, 1, 2])
        observed = torch.tensor([0, 1, 2, 3, 4, 0])
        first_pool.observed_y = observed.clone()
        second_pool.observed_y = observed.clone()

        first, _ = inject_symmetric_pseudo_label_noise(first_pool, 0.5, seed=9)
        second, _ = inject_symmetric_pseudo_label_noise(second_pool, 0.5, seed=9)

        torch.testing.assert_close(first.observed_y, second.observed_y)

    def test_hidden_annotations_do_not_change_memory_selection(self):
        generator = torch.Generator().manual_seed(8)
        x = torch.randn((15, 8, 3000), generator=generator).to(torch.float16)
        observed = torch.arange(5).repeat_interleave(3)
        first_pool = EpochPool(
            x=x.clone(),
            observed_y=observed.clone(),
            true_y=observed.clone(),
            subject_y=torch.full((15,), 64, dtype=torch.long),
        )
        second_pool = EpochPool(
            x=x.clone(),
            observed_y=observed.clone(),
            true_y=(observed + 1) % 5,
            subject_y=torch.full((15,), 64, dtype=torch.long),
        )
        first_model = CompactEEGClassifier()
        second_model = copy.deepcopy(first_model)
        first_memory = PuriDivERMemory(capacity=5, seed=12)
        second_memory = PuriDivERMemory(capacity=5, seed=12)

        first_memory.update(first_pool, first_model, torch.device("cpu"), 8, 0.25)
        second_memory.update(second_pool, second_model, torch.device("cpu"), 8, 0.25)

        torch.testing.assert_close(first_memory.pool.x, second_memory.pool.x)
        torch.testing.assert_close(
            first_memory.pool.observed_y, second_memory.pool.observed_y
        )

    def test_hidden_annotations_do_not_change_cru_partition(self):
        generator = torch.Generator().manual_seed(21)
        x = torch.randn((20, 8, 3000), generator=generator).to(torch.float16)
        observed = torch.arange(5).repeat(4)
        first_pool = EpochPool(
            x=x.clone(),
            observed_y=observed.clone(),
            true_y=observed.clone(),
            subject_y=torch.full((20,), 64, dtype=torch.long),
        )
        second_pool = EpochPool(
            x=x.clone(),
            observed_y=observed.clone(),
            true_y=(observed + 1) % 5,
            subject_y=torch.full((20,), 64, dtype=torch.long),
        )
        model = CompactEEGClassifier()

        first = puridiver_split(model, first_pool, torch.device("cpu"), 8, seed=5)
        second = puridiver_split(model, second_pool, torch.device("cpu"), 8, seed=5)

        torch.testing.assert_close(first.clean_probability, second.clean_probability)
        torch.testing.assert_close(
            first.low_uncertainty_probability,
            second.low_uncertainty_probability,
        )
        torch.testing.assert_close(first.clean_mask, second.clean_mask)
        torch.testing.assert_close(first.relabel_mask, second.relabel_mask)
        torch.testing.assert_close(first.unlabeled_mask, second.unlabeled_mask)

    def test_brainuicl_metric_summaries_match_reference_formulas(self):
        stability = {
            "ACC": [0.5, 0.4],
            "MF1": [0.3, 0.4],
            "AAA": [0.5, 0.45],
            "AAF1": [0.3, 0.35],
            "FR": [0.0, 0.2],
        }
        plasticity = {
            "1": {"ACC": [0.2, 0.3, 0.5], "MF1": [0.1, 0.2, 0.4]},
            "2": {"ACC": [0.4, 0.5, 0.6], "MF1": [0.3, 0.4, 0.5]},
        }

        stability_summary = brainuicl_stability_summary(stability)
        plasticity_summary = brainuicl_plasticity_summary(plasticity)

        self.assertAlmostEqual(stability_summary["acc"], 0.4)
        self.assertAlmostEqual(stability_summary["aaa"], 0.45)
        self.assertAlmostEqual(stability_summary["aaf1"], 0.35)
        self.assertAlmostEqual(stability_summary["fr"], 0.2)
        self.assertAlmostEqual(stability_summary["old_acc_change"], -0.1)
        self.assertAlmostEqual(stability_summary["old_mf1_change"], 0.1)
        self.assertAlmostEqual(stability_summary["relative_old_acc_change"], -0.2)
        self.assertAlmostEqual(plasticity_summary["initial_acc"], 0.3)
        self.assertAlmostEqual(plasticity_summary["before_acc"], 0.4)
        self.assertAlmostEqual(plasticity_summary["after_acc"], 0.55)

    def test_brainuicl_fr_is_not_mislabeled_when_random_model_improves(self):
        stability = {
            "ACC": [0.25, 0.50],
            "MF1": [0.10, 0.40],
            "AAA": [0.25, 0.375],
            "AAF1": [0.10, 0.25],
            "FR": [0.0, 1.0],
        }

        summary = brainuicl_stability_summary(stability)

        # BrainUICL's historical FR field is an absolute endpoint change. The
        # signed fields make clear that this case is learning, not forgetting.
        self.assertAlmostEqual(summary["fr"], 1.0)
        self.assertAlmostEqual(summary["endpoint_relative_abs_change"], 1.0)
        self.assertAlmostEqual(summary["old_acc_change"], 0.25)
        self.assertAlmostEqual(summary["relative_old_acc_change"], 1.0)

    def test_brainuicl_endpoint_change_handles_zero_initial_accuracy(self):
        stability = {
            "ACC": [0.0, 0.5],
            "MF1": [0.0, 0.4],
            "AAA": [0.0, 0.25],
            "AAF1": [0.0, 0.2],
            "FR": [None, None],
        }

        summary = brainuicl_stability_summary(stability)

        self.assertIsNone(summary["fr"])
        self.assertIsNone(summary["endpoint_relative_abs_change"])

    def test_source_old_and_new_subject_splits_are_disjoint(self):
        split = reference_subject_split(list(range(1, 99)), seed=4321)
        groups = [set(split[name]) for name in ("train", "val", "old_generalization", "new_order")]

        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1 :]:
                self.assertTrue(left.isdisjoint(right))

    def test_cached_guide_must_match_current_source_split(self):
        model = CompactEEGClassifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "guide.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "metadata": {
                        "source_train_subjects": [10, 11],
                        "source_val_subjects": [12],
                    },
                },
                checkpoint,
            )
            args = type(
                "GuideArgs",
                (),
                {
                    "device": torch.device("cpu"),
                    "guide_checkpoint": checkpoint,
                    "retrain_guide": False,
                },
            )()

            with self.assertRaisesRegex(RuntimeError, "source split"):
                train_or_load_guiding_model(args, [1, 2], [3])

    def test_student_can_start_from_guide_or_random_weights(self):
        guide = CompactEEGClassifier()
        args = InitializationArgs()
        copied = initialize_student(guide, args)
        args.student_initialization = "random"
        random_student = initialize_student(guide, args)

        for guide_parameter, copied_parameter in zip(
            guide.parameters(), copied.parameters()
        ):
            torch.testing.assert_close(guide_parameter, copied_parameter)
        self.assertTrue(
            any(
                not torch.equal(guide_parameter, random_parameter)
                for guide_parameter, random_parameter in zip(
                    guide.parameters(), random_student.parameters()
                )
            )
        )

    def test_cpc_updates_only_guide_encoder_without_label_files(self):
        guide = CompactEEGClassifier()
        encoder_before = [parameter.detach().clone() for parameter in guide.encoder.parameters()]
        classifier_before = [
            parameter.detach().clone() for parameter in guide.classifier.parameters()
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = []
            for index in range(4):
                path = Path(temporary_directory) / f"{index}.npy"
                np.save(path, np.random.default_rng(index).normal(size=(20, 8, 3000)).astype(np.float32))
                paths.append(path)
            diagnostics = adapt_guiding_model_cpc(guide, paths, CpcArgs(), task_index=1)

        self.assertGreater(diagnostics["updates"], 0)
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(encoder_before, guide.encoder.parameters())
            )
        )
        for before, after in zip(classifier_before, guide.classifier.parameters()):
            torch.testing.assert_close(before, after)


if __name__ == "__main__":
    unittest.main()
