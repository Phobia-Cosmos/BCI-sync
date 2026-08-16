import unittest

import torch

from experiments.raeeg_lop_probe import effective_rank, parse_int_list


class RAEEGLoPProbeTest(unittest.TestCase):
    def test_integer_lists_are_sorted_and_deduplicated(self):
        self.assertEqual(parse_int_list("25,0,10,10"), [0, 10, 25])

    def test_effective_rank_reports_known_rank_one_matrix(self):
        direction = torch.tensor([[1.0, 2.0, 3.0]])
        scale = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        summary = effective_rank(scale @ direction)
        self.assertAlmostEqual(summary["effective_rank"], 1.0, places=5)
        self.assertAlmostEqual(summary["stable_rank"], 1.0, places=5)
        self.assertEqual(summary["rank90"], 1)


if __name__ == "__main__":
    unittest.main()
