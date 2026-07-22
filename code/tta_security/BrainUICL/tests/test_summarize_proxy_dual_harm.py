from __future__ import annotations

import copy
import unittest

from experiments.summarize_proxy_dual_harm import (
    ATTACK_CONFIG,
    COMMON_CONFIG,
    METRIC_KEYS,
    METHOD_REGULARIZER_CONFIG,
    PROTOCOL_INVARIANTS,
    attack_diagnostics,
    benign_repeat_diagnostics,
    validate_benign_repeat,
    validate_run,
)


def _summary_and_endpoints():
    summary = {key: 0.1 for key in METRIC_KEYS}
    summary.update(
        {
            "final_old_acc": 0.5,
            "final_old_mf1": 0.4,
            "final_seen_acc": 0.6,
            "final_seen_mf1": 0.5,
        }
    )
    initial = {
        name: {"acc": 0.7, "mf1": 0.6, "n_epochs": 10}
        for name in ("old_generalization", "source_train", "validation")
    }
    final = {
        "old_generalization": {"acc": 0.5, "mf1": 0.4, "n_epochs": 10},
        "seen_subjects": {
            str(subject): {"acc": 0.6, "mf1": 0.5, "n_epochs": 10}
            for subject in range(101, 150)
        },
    }
    return summary, initial, final


def synthetic_artifacts():
    attack_tasks = list(range(1, 50, 2))
    config = dict(COMMON_CONFIG)
    config.update(
        {
            "data_root": "/data",
            "input_checkpoint_root": "/checkpoint",
            **METHOD_REGULARIZER_CONFIG["ewc"],
            **ATTACK_CONFIG,
        }
    )
    split = {"new_order": list(range(101, 150))}
    tasks = []
    for task, subject in enumerate(split["new_order"], start=1):
        attacked = task in attack_tasks
        total = 4
        attack = None
        if attacked:
            attack = {
                "mode": "proxy_dual_harm",
                "task": task,
                "subject": subject,
                "poisoned_sequences": total,
                "total_sequences": total,
                "poison_fraction": 1.0,
                "poison_indices": list(range(total)),
                "generated_inputs": True,
                "learner_replay": False,
                "injected_proxy_sequences": 3 * total,
                "training_sequences_after_injection": 4 * total,
                "proxy_repeat": 3,
                "materialized_files_deleted": total,
                "diagnostics_mean": {
                    "epsilon_eog": 1.0,
                    "epsilon_eeg": 1.0,
                    "linf_eog": 0.5,
                    "linf_eeg": 0.5,
                    "relative_l2_eog": 0.2,
                    "relative_l2_eeg": 0.2,
                    "objective_initial": 2.0,
                    "objective_final": 1.0,
                    "gradient_conflict_final": 0.1,
                    "virtual_old_loss_final": 0.2,
                    "virtual_new_loss_final": 0.3,
                    "target_shift": 1.0,
                    "target_shift_conflict": -0.2,
                    "target_hit_rate": 0.4,
                    "pseudo_label_preservation": 0.25,
                    "guiding_confidence": 0.9,
                },
            }
        tasks.append(
            {
                "task": task,
                "subject": subject,
                "current_before": {"acc": 0.5, "mf1": 0.4, "n_epochs": 10},
                "current_after": {"acc": 0.5, "mf1": 0.4, "n_epochs": 10},
                "pseudo_labels": {
                    "acc_diagnostic_only": 0.2,
                    "mf1_diagnostic_only": 0.1,
                },
                "pseudo_labels_on_clean_current": {
                    "acc_diagnostic_only": 0.3,
                    "mf1_diagnostic_only": 0.2,
                },
                "attack": attack,
            }
        )
    summary, initial, final = _summary_and_endpoints()
    metrics = {
        "method": "ewc",
        "protocol": {
            **PROTOCOL_INVARIANTS,
            "attack": "proxy_dual_harm",
            "attacker_reference_inputs_used_by_learner": False,
        },
        "tasks": tasks,
        "initial": initial,
        "final": final,
        "summary": summary,
    }
    return config, split, metrics


def synthetic_benign_artifacts():
    config, split, metrics = synthetic_artifacts()
    config.update(
        {
            "attack_mode": "benign_repeat",
            "attack_fraction": 0.05,
            "attack_eps_scale": 0.1,
            "attack_max_relative_l2": 0.0,
        }
    )
    metrics["protocol"] = {
        **PROTOCOL_INVARIANTS,
        "attack": "benign_repeat",
        "attacker_reference_inputs_used_by_learner": False,
    }
    for row in metrics["tasks"]:
        if row["attack"] is None:
            continue
        row["attack"].update(
            {
                "mode": "benign_repeat",
                "poisoned_sequences": 0,
                "poison_fraction": 0.0,
                "generated_inputs": False,
            }
        )
        row["attack"]["diagnostics_mean"] = {
            "relative_l2_eog": 0.0,
            "relative_l2_eeg": 0.0,
        }
    return config, split, metrics


class ProxyDualHarmSummaryTests(unittest.TestCase):
    def test_validates_complete_protocol_and_aggregates_diagnostics(self):
        config, split, metrics = synthetic_artifacts()
        attacked = validate_run("ewc", config, split, metrics)
        self.assertEqual(len(attacked), 25)
        summary = attack_diagnostics(attacked)
        self.assertEqual(summary["generated_sequences"], 100)
        self.assertEqual(summary["injected_proxy_copies"], 300)
        self.assertEqual(summary["materialized_files_deleted"], 100)
        self.assertAlmostEqual(summary["mean_proxy_pseudo_preservation"], 0.25)

    def test_rejects_incomplete_or_mismatched_runs(self):
        config, split, metrics = synthetic_artifacts()
        metrics["tasks"].pop()
        with self.assertRaises(RuntimeError):
            validate_run("ewc", config, split, metrics)

        config, split, metrics = synthetic_artifacts()
        config["attack_proxy_repeat"] = 2
        with self.assertRaises(RuntimeError):
            validate_run("ewc", config, split, metrics)

    def test_validates_volume_matched_clean_repeat(self):
        config, split, metrics = synthetic_benign_artifacts()
        controlled = validate_benign_repeat("ewc", config, split, metrics)
        self.assertEqual(len(controlled), 25)
        self.assertEqual(
            sum(row["attack"]["injected_proxy_sequences"] for row in controlled),
            300,
        )
        diagnostics = benign_repeat_diagnostics(controlled)
        self.assertAlmostEqual(
            diagnostics[
                "mean_clean_current_pseudo_acc_after_repeated_clean_cpc"
            ],
            0.3,
        )

    def test_rejects_nonzero_benign_repeat_perturbation_or_wrong_volume(self):
        config, split, metrics = synthetic_benign_artifacts()
        metrics["tasks"][0]["attack"]["diagnostics_mean"][
            "relative_l2_eeg"
        ] = 0.01
        with self.assertRaises(RuntimeError):
            validate_benign_repeat("ewc", config, split, metrics)

        config, split, metrics = synthetic_benign_artifacts()
        metrics["tasks"][0]["attack"]["training_sequences_after_injection"] = 3
        with self.assertRaises(RuntimeError):
            validate_benign_repeat("ewc", config, split, metrics)

    def test_rejects_attack_budget_or_label_protocol_violation(self):
        config, split, metrics = synthetic_artifacts()
        metrics["tasks"][0]["attack"]["diagnostics_mean"]["relative_l2_eeg"] = 0.21
        with self.assertRaises(RuntimeError):
            validate_run("ewc", config, split, metrics)

        config, split, metrics = synthetic_artifacts()
        metrics["protocol"]["true_target_labels_used_for_training"] = True
        with self.assertRaises(RuntimeError):
            validate_run("ewc", config, split, metrics)


if __name__ == "__main__":
    unittest.main()
