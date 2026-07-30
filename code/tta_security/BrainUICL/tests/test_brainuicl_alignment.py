import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.rttdp_brainuicl_full import (
    BufferDataset,
    parse_int_set,
    same_batch_cea_loss,
    summarize_plasticity,
)


class BrainUICLAlignmentTest(unittest.TestCase):
    def test_parse_int_set(self):
        self.assertEqual(parse_int_set("10,25,49"), {10, 25, 49})
        self.assertEqual(parse_int_set(""), set())

    def test_buffer_dataset_uses_faced_1_31_channel_interface(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            new_data = root / "new_data.npy"
            new_label = root / "new_label.npy"
            replay_data = root / "replay_data.npy"
            replay_label = root / "replay_label.npy"
            np.save(new_data, np.zeros((20, 32, 8), dtype=np.float32))
            np.save(new_label, np.zeros(20, dtype=np.int64))
            np.save(replay_data, np.ones((20, 32, 8), dtype=np.float32))
            np.save(replay_label, np.ones(20, dtype=np.int64))

            dataset = BufferDataset(
                ([new_data], [new_label]),
                ([replay_data], [replay_label]),
                train_len=1,
            )
            eog, eeg, labels = dataset[0]

            self.assertEqual(tuple(eog.shape), (40, 1, 8))
            self.assertEqual(tuple(eeg.shape), (40, 31, 8))
            self.assertEqual(tuple(labels.shape), (40,))

    def test_aligned_summary_uses_final_seen_evaluation_for_bwt(self):
        performance = {
            "plasticity": {
                "1": {"ACC": [0.5, 0.6, 0.8], "MF1": [0.4, 0.5, 0.7]},
                "2": {"ACC": [0.6, 0.7, 0.6], "MF1": [0.5, 0.6, 0.5]},
            },
            "stability": {
                "ACC": [0.75, 0.70],
                "MF1": [0.70, 0.65],
                "AAA": [0.75, 0.725],
                "AAF1": [0.70, 0.675],
                "FR": [0.0, 0.05],
            },
            "buffer": [
                {
                    "accepted_sequences": 2,
                    "candidate_sequences": 4,
                    "total_sequences": 8,
                    "length": 10,
                },
                {
                    "accepted_sequences": 3,
                    "candidate_sequences": 6,
                    "total_sequences": 12,
                    "length": 13,
                },
            ],
            "final": {
                "seen_subjects": {
                    "1": {"acc": 0.7, "mf1": 0.6},
                    "2": {"acc": 0.65, "mf1": 0.55},
                }
            },
        }
        summary = summarize_plasticity(performance)
        self.assertAlmostEqual(summary["final_seen_acc"], 0.675)
        self.assertAlmostEqual(summary["final_seen_mf1"], 0.575)
        self.assertAlmostEqual(summary["bwt_acc"], -0.025)
        self.assertAlmostEqual(summary["bwt_mf1"], -0.025)
        self.assertAlmostEqual(summary["high_confidence_candidate_rate"], 0.5)
        self.assertAlmostEqual(summary["candidate_acceptance_rate"], 0.5)
        self.assertAlmostEqual(summary["pseudo_sequence_coverage"], 0.25)
        self.assertEqual(summary["final_buffer_sequences"], 13)

    def test_snapshot_cea_compares_the_same_input_batch(self):
        class FeatureBlock(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(float(scale)))

            def forward(self, eeg, eog):
                values = torch.cat((eog, eeg), dim=1).mean(dim=-1)
                return (values * self.scale).unsqueeze(1)

        args = SimpleNamespace(
            model_param=SimpleNamespace(EogNum=1, EegNum=1, EpochLength=4)
        )
        eog = torch.zeros((1, 1, 1, 4))
        eeg = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
        reference = (FeatureBlock(1.0), nn.Identity())
        identical = (FeatureBlock(1.0), nn.Identity())
        shifted = (FeatureBlock(2.0), nn.Identity())

        identical_loss = same_batch_cea_loss(
            identical,
            reference,
            eog,
            eeg,
            args,
        )
        shifted_loss = same_batch_cea_loss(
            shifted,
            reference,
            eog,
            eeg,
            args,
        )

        self.assertAlmostEqual(float(identical_loss.detach()), 0.0, places=7)
        self.assertGreater(float(shifted_loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
