import unittest
from pathlib import Path

from experiments.spr_eeg_random_init.protocols import shuffled_subject_order, split_sequence_paths


class SprRandomInitProtocolTest(unittest.TestCase):
    def test_sequence_holdout_is_reproducible_disjoint_and_exhaustive(self):
        paths = (
            [Path(f"data/{index}.npy") for index in range(10)],
            [Path(f"label/{index}.npy") for index in range(10)],
        )
        train, test = split_sequence_paths(paths, holdout_ratio=0.2, seed=12)
        repeated_train, repeated_test = split_sequence_paths(paths, holdout_ratio=0.2, seed=12)
        self.assertEqual(train, repeated_train)
        self.assertEqual(test, repeated_test)
        self.assertEqual(len(train[0]), 8)
        self.assertEqual(len(test[0]), 2)
        self.assertTrue(set(train[0]).isdisjoint(test[0]))
        self.assertEqual(set(train[0]) | set(test[0]), set(paths[0]))

    def test_subject_order_uses_every_subject_once(self):
        subjects = list(range(1, 21))
        order = shuffled_subject_order(subjects, seed=4321)
        self.assertEqual(order, shuffled_subject_order(subjects, seed=4321))
        self.assertEqual(sorted(order), subjects)
        self.assertNotEqual(order, subjects)


if __name__ == "__main__":
    unittest.main()
