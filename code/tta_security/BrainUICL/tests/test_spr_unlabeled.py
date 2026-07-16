import inspect
import unittest
from pathlib import Path

import numpy as np

from experiments.spr_eeg_unlabeled.filtering import filter_pseudo_labeled_epochs


class SprUnlabeledFilterTest(unittest.TestCase):
    def test_filter_has_no_confidence_gate_and_considers_every_epoch(self):
        self.assertNotIn("confidence", inspect.signature(filter_pseudo_labeled_epochs).parameters)
        features = np.ones((2, 4, 3), dtype=np.float32)
        pseudo_labels = np.zeros((2, 4), dtype=np.int64)
        result = filter_pseudo_labeled_epochs(
            features,
            pseudo_labels,
            [Path("a.npy"), Path("b.npy")],
            ensembles=2,
            bmm_iters=2,
            seed=4,
        )
        self.assertEqual(result.metrics["candidate_epochs"], 8)
        self.assertEqual(result.metrics["accepted_epochs"], 8)

    def test_diagnostic_ground_truth_cannot_change_selection(self):
        rng = np.random.default_rng(8)
        features = rng.normal(size=(4, 5, 6)).astype(np.float32)
        pseudo_labels = np.arange(20, dtype=np.int64).reshape(4, 5) % 5
        paths = [Path(f"{index}.npy") for index in range(4)]
        first = filter_pseudo_labeled_epochs(
            features,
            pseudo_labels,
            paths,
            ensembles=2,
            bmm_iters=2,
            seed=7,
            true_labels_for_diagnostics=pseudo_labels,
        )
        second = filter_pseudo_labeled_epochs(
            features,
            pseudo_labels,
            paths,
            ensembles=2,
            bmm_iters=2,
            seed=7,
            true_labels_for_diagnostics=(pseudo_labels + 1) % 5,
        )
        np.testing.assert_array_equal(first.accepted, second.accepted)
        self.assertEqual(
            [(record.data_path, record.epoch_index, record.observed_label) for record in first.records],
            [(record.data_path, record.epoch_index, record.observed_label) for record in second.records],
        )


if __name__ == "__main__":
    unittest.main()
