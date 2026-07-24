from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np

from model.full_spr_eeg import (
    SPRDelayedBuffer,
    SPRDelayedRecord,
    SPRPurifiedMemory,
    self_centered_admission,
)


def delayed(
    index: int,
    labels: np.ndarray | None = None,
    *,
    is_proxy: bool = False,
) -> SPRDelayedRecord:
    if labels is None:
        labels = np.arange(20, dtype=np.int64) % 5
    return SPRDelayedRecord(
        data_path=Path(f"{index}.npy"),
        pseudo_labels=labels,
        task=1,
        subject=64,
        sequence_index=index,
        is_proxy=is_proxy,
    )


class FullSPREEGCoreTests(unittest.TestCase):
    def test_delayed_buffer_flush_and_state_roundtrip(self):
        buffer = SPRDelayedBuffer(capacity_sequences=2)
        self.assertFalse(buffer.add(delayed(0)))
        self.assertTrue(buffer.add(delayed(1, is_proxy=True)))
        state = buffer.state_dict()
        restored = SPRDelayedBuffer.from_state_dict(state)
        self.assertEqual(len(restored), 2)
        self.assertTrue(restored.records[1].is_proxy)
        drained = restored.drain()
        self.assertEqual([record.sequence_index for record in drained], [0, 1])
        self.assertEqual(len(restored), 0)

    def test_purified_memory_is_epoch_capacity_and_evicts_low_probability(self):
        memory = SPRPurifiedMemory(capacity_epochs=3, num_classes=5, seed=7)
        source = delayed(0, labels=np.zeros(5, dtype=np.int64))
        probabilities = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
        update = memory.add(
            [source],
            [np.ones(5, dtype=bool)],
            [probabilities],
        )
        self.assertEqual(len(memory), 3)
        self.assertEqual(update["evicted_epochs"], 2)
        kept = memory.records[0].clean_probabilities[memory.records[0].epoch_mask]
        self.assertEqual(sorted(kept.tolist()), [0.7, 0.8, 0.9])

    def test_purified_memory_state_preserves_masks_rng_and_proxy_tracking(self):
        memory = SPRPurifiedMemory(capacity_epochs=20, num_classes=5, seed=11)
        source = delayed(3, is_proxy=True)
        mask = np.arange(20) % 2 == 0
        memory.add([source], [mask], [np.linspace(0.1, 0.9, 20)])
        restored = SPRPurifiedMemory.from_state_dict(memory.state_dict())
        self.assertEqual(restored.stats()["proxy_epochs"], 10)
        first = [record.sequence_index for record in memory.sample_records(4)]
        second = [record.sequence_index for record in restored.sample_records(4)]
        self.assertEqual(first, second)
        self.assertEqual(memory.stats(), restored.stats())

    def test_self_centered_admission_is_deterministic_and_label_isolated(self):
        records = [
            delayed(0, labels=np.zeros(20, dtype=np.int64)),
            delayed(1, labels=np.ones(20, dtype=np.int64)),
        ]
        features = np.random.default_rng(3).normal(size=(2, 20, 8)).astype(np.float32)
        first = self_centered_admission(
            features,
            records,
            ensembles=2,
            bmm_iters=2,
            graph_seed=5,
            admission_rng=np.random.default_rng(9),
        )
        second = self_centered_admission(
            features,
            records,
            ensembles=2,
            bmm_iters=2,
            graph_seed=5,
            admission_rng=np.random.default_rng(9),
        )
        self.assertTrue(np.array_equal(np.stack(first[0]), np.stack(second[0])))
        self.assertTrue(np.allclose(np.stack(first[1]), np.stack(second[1])))
        parameters = set(inspect.signature(self_centered_admission).parameters)
        self.assertNotIn("true_labels", parameters)
        self.assertNotIn("label_paths", parameters)


if __name__ == "__main__":
    unittest.main()
