import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from pure_puridiver_eeg import (  # noqa: E402
    CompactEEGClassifier,
    EpochPool,
    PuriDivERMemory,
    fit_low_component_probability,
    split_subject_paths,
    symmetric_noise,
)


class PurePuriDivEREEGTest(unittest.TestCase):
    def test_two_gmm_means_match_separated_loss_groups(self):
        low = np.array([0.08, 0.12, 0.15, 0.18] * 4)
        high = np.array([0.76, 0.91] * 8)
        probability, means = fit_low_component_probability(
            np.concatenate((low, high)), seed=9
        )

        self.assertAlmostEqual(means[0], low.mean(), places=2)
        self.assertAlmostEqual(means[1], high.mean(), places=2)
        self.assertGreater(probability[: low.size].mean(), 0.95)
        self.assertLess(probability[low.size :].mean(), 0.05)

    def test_observed_labels_equal_true_labels_only_on_clean_stream(self):
        true_labels = np.tile(np.arange(5), 100)
        clean, clean_mask = symmetric_noise(true_labels, rate=0.0, seed=3)
        noisy, noisy_mask = symmetric_noise(true_labels, rate=0.4, seed=3)

        np.testing.assert_array_equal(clean, true_labels)
        self.assertFalse(clean_mask.any())
        self.assertTrue(noisy_mask.any())
        self.assertTrue(np.all(noisy[noisy_mask] != true_labels[noisy_mask]))

    def test_subject_sequence_split_is_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_directory = root / "7" / "data"
            data_directory.mkdir(parents=True)
            for index in range(10):
                (data_directory / f"{index}.npy").touch()

            train, test = split_subject_paths(root, subject=7, train_fraction=0.7)

        self.assertEqual(len(train), 7)
        self.assertEqual(len(test), 3)
        self.assertTrue(set(train).isdisjoint(test))

    def test_dynamic_memory_selection_respects_capacity_and_class_balance(self):
        labels = torch.arange(5).repeat_interleave(3)
        pool = EpochPool(
            x=torch.randn((15, 8, 3000), dtype=torch.float16),
            observed_y=labels.clone(),
            true_y=labels.clone(),
            subject_y=torch.ones(15, dtype=torch.long),
        )
        memory = PuriDivERMemory(capacity=5, seed=4)
        model = CompactEEGClassifier()

        diagnostics = memory.update(
            pool,
            model,
            device=torch.device("cpu"),
            infer_batch_size=8,
            diversity_coefficient=0.25,
        )

        self.assertEqual(len(memory), 5)
        self.assertEqual(diagnostics["removed"], 10)
        self.assertEqual(
            torch.bincount(memory.pool.observed_y, minlength=5).tolist(),
            [1, 1, 1, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
