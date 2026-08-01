import unittest

import numpy as np

from experiments.persist_eeg import DirectionBank, ProbabilityStateFilter, proxy_information_score


class PersistEegTest(unittest.TestCase):
    def test_probability_filter_tracks_feedback_without_parameters(self):
        first = np.full((2, 3, 4), 0.25, dtype=np.float32)
        second = first.copy()
        second[..., 0] = 0.7
        second[..., 1:] = 0.1
        state = ProbabilityStateFilter()
        first_row = state.update(first)
        second_row = state.update(second)
        self.assertEqual(first_row["observations"], 1.0)
        self.assertEqual(second_row["observations"], 2.0)
        self.assertGreater(second_row["kl_to_long"], 0.0)
        self.assertLess(second_row["mean_margin"], 1.0)

    def test_direction_bank_is_bounded_and_reports_alignment(self):
        bank = DirectionBank(capacity=2, decay=0.5)
        direction = np.ones((2, 3), dtype=np.float32)
        bank.update(direction)
        bank.update(direction)
        bank.update(-np.ones((5, 3), dtype=np.float32))
        self.assertEqual(len(bank.directions), 2)
        self.assertIsNotNone(bank.cosine(direction))
        self.assertEqual(bank.state()["bank_size"], 2)

    def test_information_score_is_finite(self):
        probabilities = np.full((2, 3, 4), 0.25, dtype=np.float32)
        result = proxy_information_score(probabilities, 0.5)
        self.assertTrue(np.isfinite(result["information_score"]))
        self.assertEqual(result["direction_bank_cosine"], 0.5)


if __name__ == "__main__":
    unittest.main()
