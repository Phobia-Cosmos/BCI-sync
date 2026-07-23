from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from experiments.aligned_replay_defenses import (
    CRUState,
    PuriDivERSequenceMemory,
    ReservoirReplayMemory,
    apply_spr_filter,
    build_cru_state,
    build_memory_records,
    collect_epoch_outputs,
    load_replay_batch,
    puridiver_branch_loss,
)


class TinyFeatureExtractor(nn.Module):
    def __init__(self, sequence_length: int, dimension: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.projection = nn.Linear(2, dimension)

    def forward(self, eeg: torch.Tensor, eog: torch.Tensor) -> torch.Tensor:
        batch = eeg.shape[0] // self.sequence_length
        summary = torch.stack(
            (eeg.mean(dim=(1, 2)), eog.mean(dim=(1, 2))), dim=1
        )
        return self.projection(summary).reshape(batch, self.sequence_length, -1)


class TinyTransformer(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(values))


class TinySleepClassifier(nn.Module):
    def __init__(self, dimension: int, classes: int):
        super().__init__()
        self.sleep_stage_mlp = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.Tanh(),
        )
        self.sleep_stage_classifier = nn.Linear(128, classes, bias=False)


def tiny_blocks(sequence_length: int = 3, classes: int = 2):
    torch.manual_seed(4)
    dimension = 5
    return (
        TinyFeatureExtractor(sequence_length, dimension),
        TinyTransformer(dimension),
        TinySleepClassifier(dimension, classes),
    )


def args(batch: int = 2):
    return SimpleNamespace(
        batch=batch,
        num_worker=0,
        device=torch.device("cpu"),
        spr_ensembles=2,
        spr_bmm_iters=3,
        puridiver_strong_noise=0.02,
        puridiver_strong_scale=0.05,
        puridiver_strong_mask_fraction=0.1,
        puridiver_consistency_weight=1.0,
    )


def write_sequence(path: Path, value: float, sequence_length: int = 3) -> None:
    signal = np.full((sequence_length, 4, 12), value, dtype=np.float32)
    signal[:, :, ::2] += 0.1
    np.save(path, signal)


class AlignedReplayDefenseTests(unittest.TestCase):
    def test_occurrence_uid_masks_and_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.npy"
            write_sequence(path, 0.2)
            paths = [path, path]
            labels = [np.array([0, 1, 0]), np.array([1, 1, 0])]
            masks = [np.array([True, False, True]), None]
            probabilities = [np.array([0.9, 0.1, 0.8]), None]
            records = build_memory_records(
                paths,
                labels,
                task_index=2,
                subject=7,
                original_count=1,
                poisoned_paths={str(path)},
                epoch_masks=masks,
                clean_probabilities=probabilities,
            )

            self.assertNotEqual(records[0].uid, records[1].uid)
            self.assertFalse(records[0].repeated_upload)
            self.assertTrue(records[1].repeated_upload)
            self.assertEqual(records[0].serializable()["epoch_mask"], [True, False, True])
            self.assertIsNone(records[1].serializable()["epoch_mask"])
            _eog, _eeg, replay_labels = load_replay_batch(records)
            self.assertEqual(replay_labels[0].tolist(), [0, -100, 0])

            memory = ReservoirReplayMemory(capacity=2, seed=3)
            memory.add(records)
            self.assertEqual(memory.stats()["retained_epochs"], 5)

    def test_collect_outputs_have_aligned_shapes_and_reject_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{index}.npy" for index in range(2)]
            for index, path in enumerate(paths):
                write_sequence(path, float(index))
            logits, epoch_embeddings, sequence_embeddings = collect_epoch_outputs(
                tiny_blocks(), paths, args()
            )

        self.assertEqual(tuple(logits.shape), (2, 2, 3))
        self.assertEqual(tuple(epoch_embeddings.shape), (2, 3, 5))
        self.assertEqual(tuple(sequence_embeddings.shape), (2, 128))
        with self.assertRaisesRegex(ValueError, "at least one signal path"):
            collect_epoch_outputs(tiny_blocks(), [], args())

    def test_spr_uses_all_epochs_drops_empty_occurrences_and_is_deterministic(self):
        labels = [np.array([0, 0, 1]), np.array([1, 0, 1])]

        def records():
            return build_memory_records(
                [Path("0.npy"), Path("1.npy")],
                labels,
                task_index=1,
                subject=3,
                original_count=2,
                poisoned_paths=set(),
            )

        embeddings = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        probabilities = np.array([0.0, 0.0, 0.0, 1.0, 0.55, 1.0])
        with patch(
            "experiments.aligned_replay_defenses.spr_eeg.self_centered_clean_probabilities",
            return_value=probabilities,
        ) as clean_probability:
            first, first_stats = apply_spr_filter(records(), embeddings, args(), seed=9)
            second, second_stats = apply_spr_filter(records(), embeddings, args(), seed=9)

        self.assertEqual(clean_probability.call_args.args[0].shape, (6, 4))
        self.assertEqual(clean_probability.call_args.args[1].shape, (6,))
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].uid, second[0].uid)
        np.testing.assert_array_equal(first[0].epoch_mask, second[0].epoch_mask)
        self.assertEqual(first_stats, second_stats)
        self.assertEqual(first_stats["dropped_sequences"], 1)

    def test_cru_second_gmm_only_sees_noisy_subset_and_masks_exhaust(self):
        logits = torch.tensor(
            [[4.0, -1.0], [3.0, -0.5], [0.4, 0.6], [0.1, 0.9]]
        )
        observed = torch.tensor([0, 0, 0, 0])
        with patch(
            "experiments.aligned_replay_defenses._low_component_probability",
            side_effect=[
                (np.array([0.9, 0.8, 0.1, 0.2]), [0.1, 1.0]),
                (np.array([0.8, 0.2]), [0.1, 0.5]),
            ],
        ) as fit_gmm:
            state = build_cru_state(logits, observed, seed=8, thresholds=(0.5, 0.5))

        self.assertEqual(fit_gmm.call_args_list[1].args[0].shape, (2,))
        self.assertEqual(state.clean_mask.tolist(), [True, True, False, False])
        self.assertEqual(state.relabel_mask.tolist(), [False, False, True, False])
        self.assertEqual(state.unlabeled_mask.tolist(), [False, False, False, True])
        total = state.clean_mask.int() + state.relabel_mask.int() + state.unlabeled_mask.int()
        self.assertTrue(total.eq(1).all())

    def test_cru_ignores_masked_replay_epochs(self):
        logits = torch.tensor(
            [[4.0, -1.0], [0.1, 0.9], [-1.0, 4.0]]
        )
        observed = torch.tensor([0, -100, 1])

        state = build_cru_state(logits, observed, seed=3)

        self.assertEqual(state.diagnostics["ignored_count"], 1)
        self.assertEqual(state.clean_mask.tolist(), [True, False, True])
        self.assertFalse(state.relabel_mask[1].item())
        self.assertFalse(state.unlabeled_mask[1].item())

    def test_puridiver_branch_loss_is_finite_and_backpropagates(self):
        blocks = tiny_blocks()
        eog = torch.randn(1, 3, 2, 12)
        eeg = torch.randn(1, 3, 2, 12)
        observed = torch.tensor([[0, 1, 0]])
        state = CRUState(
            clean_probability=torch.tensor([[0.9, 0.1, 0.1]]),
            low_uncertainty_probability=torch.tensor([[0.0, 0.7, 0.1]]),
            snapshot_probabilities=torch.tensor(
                [[[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]]
            ),
            clean_mask=torch.tensor([[True, False, False]]),
            relabel_mask=torch.tensor([[False, True, False]]),
            unlabeled_mask=torch.tensor([[False, False, True]]),
            diagnostics={},
        )

        loss, stats = puridiver_branch_loss(
            blocks, eog, eeg, observed, state, args()
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            (stats["clean_count"], stats["relabel_count"], stats["unlabeled_count"]),
            (1, 1, 1),
        )
        gradients = [
            parameter.grad for block in blocks for parameter in block.parameters()
        ]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
        )

    def test_puridiver_sequence_memory_caps_and_repeats_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{index}.npy" for index in range(6)]
            for index, path in enumerate(paths):
                write_sequence(path, index / 10.0)
            labels = [np.full(3, index % 2, dtype=np.int64) for index in range(6)]

            def records():
                return build_memory_records(
                    paths,
                    labels,
                    task_index=1,
                    subject=4,
                    original_count=6,
                    poisoned_paths=set(),
                )

            first = PuriDivERSequenceMemory(capacity=3, seed=12)
            second = PuriDivERSequenceMemory(capacity=3, seed=12)
            first_update = first.add(records(), tiny_blocks(), args(), 0.4)
            second_update = second.add(records(), tiny_blocks(), args(), 0.4)

            self.assertEqual(len(first), 3)
            self.assertEqual(first_update["removed"], 3)
            self.assertEqual(first_update, second_update)
            self.assertEqual(
                [record.uid for record in first.records],
                [record.uid for record in second.records],
            )
            sampled = first.sample(5)
            self.assertEqual(len(sampled), 5)
            self.assertEqual(first.stats()["total_replay_draws"], 5)
            self.assertEqual(len(first.serializable_records()), 3)


if __name__ == "__main__":
    unittest.main()
