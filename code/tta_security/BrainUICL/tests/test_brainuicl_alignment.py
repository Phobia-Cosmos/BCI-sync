import unittest

from experiments.rttdp_brainuicl_full import parse_int_set, summarize_plasticity


class BrainUICLAlignmentTest(unittest.TestCase):
    def test_parse_int_set(self):
        self.assertEqual(parse_int_set("10,25,49"), {10, 25, 49})
        self.assertEqual(parse_int_set(""), set())

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


if __name__ == "__main__":
    unittest.main()
