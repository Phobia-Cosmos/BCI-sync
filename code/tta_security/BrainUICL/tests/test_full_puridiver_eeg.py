from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np

from model.full_puridiver_eeg import (
    DynamicPuriMemory,
    PuriMemoryScores,
    PuriSequenceRecord,
    build_cru_partition,
)


def record(index: int, labels: np.ndarray, *, is_proxy: bool = False):
    return PuriSequenceRecord(
        data_path=Path(f"sequence-{index}.npy"),
        pseudo_labels=labels,
        epoch_mask=np.ones(labels.size, dtype=bool),
        task=1,
        subject=64,
        sequence_index=index,
        is_proxy=is_proxy,
    )


class FullPuriDivEREEGCoreTests(unittest.TestCase):
    def test_dynamic_memory_recomputes_after_every_required_deletion(self):
        memory = DynamicPuriMemory(capacity_epochs=4, num_classes=5, seed=3)
        incoming = [
            record(0, np.array([0, 0, 1])),
            record(1, np.array([0, 1, 1]), is_proxy=True),
        ]
        scorer_calls = 0

        def scorer(records):
            nonlocal scorer_calls
            scorer_calls += 1
            count = sum(item.retained_epochs for item in records)
            return PuriMemoryScores(
                losses=np.linspace(0.1, 0.9, count),
                features=np.arange(count * 4, dtype=np.float64).reshape(count, 4) + 1,
                classifier_weights=np.arange(20, dtype=np.float64).reshape(5, 4),
            )

        update = memory.update(incoming, scorer, diversity_coefficient=0.4)
        self.assertEqual(scorer_calls, 1)
        self.assertEqual(update["removed_epochs"], 2)
        self.assertEqual(update["score_recomputations"], 2)
        self.assertEqual(len(memory), 4)
        self.assertLessEqual(max(memory.class_counts()), 2)

    def test_epoch_sampling_is_exact_and_state_roundtrips(self):
        memory = DynamicPuriMemory(capacity_epochs=20, num_classes=5, seed=11)
        incoming = [record(0, np.arange(5) % 5, is_proxy=True)]
        memory.update(
            incoming,
            lambda _records: (_ for _ in ()).throw(AssertionError("no scorer needed")),
            diversity_coefficient=0.4,
        )
        restored = DynamicPuriMemory.from_state_dict(memory.state_dict())
        sampled = memory.sample_epoch_weights(12)
        sampled_restored = restored.sample_epoch_weights(12)
        self.assertEqual(sum(int(weights.sum()) for _record, weights in sampled), 12)
        self.assertEqual(
            [weights.tolist() for _record, weights in sampled],
            [weights.tolist() for _record, weights in sampled_restored],
        )
        self.assertEqual(memory.stats(), restored.stats())

    def test_cru_partition_is_exhaustive_and_recomputed_from_two_gmms(self):
        losses = np.concatenate(
            [np.linspace(0.05, 0.15, 12), np.linspace(1.5, 2.0, 12)]
        )
        probabilities = np.zeros((24, 5), dtype=np.float64)
        probabilities[:12, 0] = 0.9
        probabilities[:12, 1:] = 0.025
        probabilities[12:18, 1] = 0.95
        probabilities[12:18, [0, 2, 3, 4]] = 0.0125
        probabilities[18:] = 0.2
        partition = build_cru_partition(
            losses, probabilities, seed=7, min_gmm_samples=4
        )
        partition.validate()
        self.assertEqual(
            int(
                partition.clean_mask.sum()
                + partition.relabel_mask.sum()
                + partition.unlabeled_mask.sum()
            ),
            24,
        )
        self.assertGreater(partition.clean_mask.sum(), 0)
        self.assertGreater(partition.relabel_mask.sum(), 0)
        self.assertGreater(partition.unlabeled_mask.sum(), 0)
        self.assertIsNotNone(partition.diagnostics["loss_gmm_means"])
        self.assertIsNotNone(partition.diagnostics["uncertainty_gmm_means"])

    def test_degenerate_loss_fallback_is_explicit_all_clean(self):
        partition = build_cru_partition(
            np.ones(5), np.full((5, 5), 0.2), seed=1, min_gmm_samples=8
        )
        self.assertTrue(partition.clean_mask.all())
        self.assertFalse(partition.relabel_mask.any())
        self.assertFalse(partition.unlabeled_mask.any())
        self.assertTrue(partition.diagnostics["loss_gmm_fallback_all_clean"])

    def test_core_interfaces_cannot_receive_target_annotations(self):
        for function in (build_cru_partition, DynamicPuriMemory.update):
            parameters = set(inspect.signature(function).parameters)
            self.assertNotIn("true_labels", parameters)
            self.assertNotIn("label_paths", parameters)


if __name__ == "__main__":
    unittest.main()
