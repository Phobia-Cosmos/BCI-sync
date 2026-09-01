import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn

from experiments.raeeg_lop_probe import (
    BrainUICLModel,
    evaluate_retention,
    effective_rank,
    parse_int_list,
    preflight_resources,
    blocked_result,
    write_markdown_report,
)


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

    def test_wrapper_returns_sequence_tokens_with_classes_last(self):
        model_param = SimpleNamespace(
            EogNum=1,
            EegNum=1,
            EpochLength=4,
            SeqLength=2,
            NumClasses=3,
        )
        args = SimpleNamespace(model_param=model_param, device=torch.device("cpu"))

        class TinyFeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(2, 2)

            def forward(self, eeg, eog):
                batch = eeg.shape[0] // model_param.SeqLength
                values = torch.cat((eeg, eog), dim=1).mean(dim=2)
                return self.projection(values).reshape(batch, model_param.SeqLength, 2)

        class TinyEncoder(nn.Module):
            def forward(self, values):
                return values

        class TinyClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.sleep_stage_mlp = nn.Linear(2, 3)

            def forward(self, values):
                return self.sleep_stage_mlp(values).permute(0, 2, 1)

        model = BrainUICLModel(
            (TinyFeatureExtractor(), TinyEncoder(), TinyClassifier()),
            args,
        )
        packed = torch.randn(1, model_param.SeqLength, 2, model_param.EpochLength)
        output = model(packed)
        self.assertEqual(tuple(output.shape), (1, model_param.SeqLength, model_param.NumClasses))

    def test_markdown_report_contains_primary_gap_and_stage_table(self):
        result = {
            "protocol": "test",
            "primary_outcome": "fresh_gap_final",
            "config": {"dataset": "ISRUC", "method": "finetune", "stages": [0], "seed": 1, "fresh_reference": "random"},
            "summary": {
                "mean_warm_acc_gain": 0.1,
                "mean_fresh_gap_final": 0.2,
                "mean_fresh_auc_gap": 0.3,
                "mean_transformer_effective_rank": 2.0,
                "transformer_effective_rank_vs_fresh_gap_pearson": None,
            },
            "tasks": [
                {
                    "stage": 0,
                    "subject": 1,
                    "spectra": {"transformer": {"effective_rank": 2.0, "effective_rank_normalized": 0.5}},
                    "plasticity": {"outcome": {"fresh_gap_final": 0.2, "fresh_auc_gap": 0.3}},
                }
            ],
        }
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_markdown_report(result, path)
            content = path.read_text(encoding="utf-8")
        self.assertIn("fresh_gap_final", content)
        self.assertIn("| 0 | 1 |", content)

    def test_retention_is_explicitly_eval_only_when_not_requested(self):
        args = SimpleNamespace()
        result = evaluate_retention(None, [], args)
        self.assertEqual(result["status"], "not-requested")
        self.assertEqual(result["label_source"], "true_labels_for_retention_eval_only")

    def test_preflight_reports_missing_external_resources_without_loading_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                data_root=root / "data",
                input_checkpoint_root=root / "parameters",
                run_root=root / "runs",
                dataset="ISRUC",
                method="finetune",
                seed=4321,
                pretrain_seed=None,
                fresh_reference="random",
                anchor_subject=-1,
                retention_subject=[],
            )
            split_path = root / "split.json"
            preflight = preflight_resources(args, split_path=split_path, stages=[0, 1])
            self.assertEqual(preflight["status"], "blocked")
            self.assertTrue(any(item["kind"] == "split-manifest" for item in preflight["blockers"]))
            blocked = blocked_result(args, preflight=preflight, stages=[0, 1], split_path=split_path)
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(blocked["block_reason"], "blocked-by-split-manifest")
            output = root / "blocked.md"
            write_markdown_report(blocked, output)
            self.assertIn("blocked-by-split-manifest", output.read_text(encoding="utf-8"))

    def test_adapter_metric_alias_can_feed_edgeforge_lagged_analyzer(self):
        from edgeforge.lop_envelope import diagnostic_to_metrics

        metrics = diagnostic_to_metrics(
            {
                "config": {"dataset": "ISRUC", "method": "finetune"},
                "tasks": [
                    {
                        "stage": 10,
                        "subject": 2,
                        "spectra": {"transformer": {"effective_rank": 7.0}},
                        "plasticity": {"outcome": {"fresh_gap_final": 0.2}},
                    }
                ],
            }
        )
        names = {item["name"] for item in metrics}
        self.assertIn("task.spectra.transformer_1.effective_rank", names)


if __name__ == "__main__":
    unittest.main()
