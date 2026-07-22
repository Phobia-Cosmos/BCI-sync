import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from model.regularization_cl import (
    QuadraticImportanceStrategy,
    SynapticIntelligenceStrategy,
    freeze_batch_norm_running_stats,
    hard_pseudo_label_loss,
    named_trainable_parameters,
)
from experiments.regularization_cl_attacks import (
    _project,
    _proxy_dual_harm_terms,
    brainwash_one_step_batch,
    materialize_batched_proxy_dual_harm_subject,
    materialize_poisoned_subject,
    pacol_gradient_matching_batch,
    proxy_dual_harm_batch,
)
from experiments.regularization_cl_eeg import (
    delete_generated_inputs,
    resolve_attack_tasks,
)
from experiments.rttdp_brainuicl_full import external_proxy_upload_paths


def tiny_blocks():
    first = nn.Linear(2, 1, bias=False)
    second = nn.Identity()
    third = nn.Identity()
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, -1.0]]))
    return first, second, third


class TinyFeatureExtractor(nn.Module):
    def forward(self, eeg, eog):
        return torch.cat((eeg.mean(dim=2), eog.mean(dim=2)), dim=1)


def tiny_proxy_blocks():
    return TinyFeatureExtractor(), nn.Identity(), nn.Linear(2, 2)


def tiny_proxy_args(**overrides):
    values = {
        "attack_param_scope": "classifier",
        "attack_curvature_scale": 1.0,
        "attack_new_proxy_weight": 1.0,
        "attack_eps_scale": 0.5,
        "attack_random_start": False,
        "attack_max_relative_l2": 0.2,
        "attack_steps": 2,
        "attack_inner_lr": 1e-2,
        "attack_min_confidence": 0.0,
        "attack_target_weight": 1.0,
        "attack_conflict_weight": 1.0,
        "attack_gradient_norm_weight": 0.1,
        "attack_virtual_old_weight": 1.0,
        "attack_virtual_new_weight": 1.0,
        "attack_confidence_weight": 0.0,
        "attack_l2_weight": 0.0,
        "attack_fraction": 1.0,
        "attack_generation_batch": 4,
        "attack_reference_batch": 1,
        "attack_mode": "proxy_dual_harm",
        "seed": 7,
        "device": torch.device("cpu"),
        "model_param": SimpleNamespace(
            EogNum=1,
            EegNum=1,
            EpochLength=2,
            NumClasses=2,
            SeqLength=2,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RegularizationCLEEGTest(unittest.TestCase):
    def test_delete_generated_inputs_stays_inside_attack_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated_root = root / "poisoned_inputs"
            generated_root.mkdir()
            generated = generated_root / "0.npy"
            generated.write_bytes(b"generated")
            original = root / "clean.npy"
            original.write_bytes(b"clean")

            self.assertEqual(
                delete_generated_inputs([generated], generated_root),
                1,
            )
            self.assertFalse(generated.exists())
            self.assertTrue(original.exists())
            with self.assertRaises(ValueError):
                delete_generated_inputs([original], generated_root)

    def test_external_proxy_upload_preserves_clean_label_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_root = root / "individual_2"
            data_dir = task_root / "data"
            data_dir.mkdir(parents=True)
            for index in range(2):
                np.save(data_dir / f"{index}.npy", np.array([index], dtype=np.float32))
            (task_root / "metadata.json").write_text(
                json.dumps({"task": 2, "subject": 89, "poisoned": 2})
            )
            clean_data = [Path("clean-0.npy"), Path("clean-1.npy")]
            clean_labels = [Path("label-0.npy"), Path("label-1.npy")]

            uploaded, metadata = external_proxy_upload_paths(
                root,
                (clean_data, clean_labels),
                task_index=2,
                subject=89,
            )

            self.assertEqual([path.name for path in uploaded[0]], ["0.npy", "1.npy"])
            self.assertEqual(uploaded[1], clean_labels)
            self.assertEqual(metadata["poisoned"], 2)

    def test_external_proxy_upload_rejects_wrong_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_root = root / "individual_1"
            data_dir = task_root / "data"
            data_dir.mkdir(parents=True)
            np.save(data_dir / "0.npy", np.array([0], dtype=np.float32))
            (task_root / "metadata.json").write_text(
                json.dumps({"task": 1, "subject": 64})
            )
            with self.assertRaises(ValueError):
                external_proxy_upload_paths(
                    root,
                    ([Path("clean.npy")], [Path("label.npy")]),
                    task_index=1,
                    subject=89,
                )

    def test_attack_interfaces_do_not_accept_true_labels(self):
        for function in (
            pacol_gradient_matching_batch,
            brainwash_one_step_batch,
            proxy_dual_harm_batch,
            materialize_poisoned_subject,
            materialize_batched_proxy_dual_harm_subject,
        ):
            parameter_names = set(inspect.signature(function).parameters)
            self.assertNotIn("labels", parameter_names)
            self.assertNotIn("true_labels", parameter_names)

    def test_projection_enforces_linf_and_per_sequence_relative_l2(self):
        base = torch.tensor(
            [
                [[[[1.0, -1.0, 2.0, -2.0]]]],
                [[[[0.5, -0.5, 1.0, -1.0]]]],
            ]
        ).reshape(2, 1, 1, 4)
        delta = torch.full_like(base, 10.0)
        projected = _project(
            delta,
            base,
            torch.tensor(0.25),
            torch.tensor(-10.0),
            torch.tensor(10.0),
            max_relative_l2=0.1,
        )
        self.assertLessEqual(float(projected.abs().max()), 0.25 + 1e-7)
        relative_l2 = torch.linalg.vector_norm(projected.flatten(1), dim=1) / (
            torch.linalg.vector_norm(base.flatten(1), dim=1) + 1e-12
        )
        self.assertTrue(bool((relative_l2 <= 0.1 + 1e-7).all()))

    def test_proxy_dual_harm_recomputes_final_diagnostics_on_returned_input(self):
        torch.manual_seed(4)
        student = tiny_proxy_blocks()
        guide = tiny_proxy_blocks()
        guide[2].load_state_dict(student[2].state_dict())
        args = tiny_proxy_args()
        eog = torch.tensor(
            [[[[0.1, 0.4]], [[0.3, -0.2]]]], dtype=torch.float32
        )
        eeg = torch.tensor(
            [[[[0.5, -0.1]], [[-0.3, 0.2]]]], dtype=torch.float32
        )
        reference_eog = eog.flip(1)
        reference_eeg = eeg.flip(1)

        with patch(
            "experiments.regularization_cl_attacks._proxy_dual_harm_terms",
            wraps=_proxy_dual_harm_terms,
        ) as terms:
            eog_adv, eeg_adv, diagnostics = proxy_dual_harm_batch(
                student,
                guide,
                eog,
                eeg,
                reference_eog,
                reference_eeg,
                args,
            )

        self.assertEqual(terms.call_count, args.attack_steps + 1)
        final_call = terms.call_args.kwargs
        self.assertTrue(torch.equal(final_call["eog_adv"], eog_adv))
        self.assertTrue(torch.equal(final_call["eeg_adv"], eeg_adv))
        self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))
        self.assertLessEqual(diagnostics["relative_l2_eog"], 0.2 + 1e-6)
        self.assertLessEqual(diagnostics["relative_l2_eeg"], 0.2 + 1e-6)

    def test_batched_materialization_weights_partial_batch_and_aligns_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_paths = []
            for index in range(5):
                path = root / f"clean-{index}.npy"
                np.save(path, np.full((2, 3, 2), index, dtype=np.float32))
                current_paths.append(path)
            args = tiny_proxy_args(attack_steps=1)

            def fake_attack(
                _student,
                _guide,
                eog,
                eeg,
                _reference_eog,
                _reference_eeg,
                _args,
                *,
                strategy,
            ):
                del strategy
                return eog, eeg, {"score": float(eog.shape[0])}

            with patch(
                "experiments.regularization_cl_attacks.proxy_dual_harm_batch",
                side_effect=fake_attack,
            ):
                mixed_paths, diagnostics = materialize_batched_proxy_dual_harm_subject(
                    student_blocks=(),
                    label_blocks=(),
                    strategy=None,
                    current_data_paths=current_paths,
                    reference_data_paths=current_paths,
                    output_dir=root / "generated",
                    task_index=1,
                    subject=64,
                    args=args,
                )

            self.assertEqual(diagnostics["generation_batch_sizes"], [4, 1])
            self.assertAlmostEqual(diagnostics["diagnostics_mean"]["score"], 3.4)
            self.assertAlmostEqual(
                diagnostics["diagnostics_batch_macro_mean"]["score"],
                2.5,
            )
            self.assertEqual([path.name for path in mixed_paths], [f"{i}.npy" for i in range(5)])
            for index, path in enumerate(mixed_paths):
                self.assertTrue(path.is_file())
                self.assertTrue(np.array_equal(np.load(path), np.load(current_paths[index])))

    def test_attack_task_resolution_supports_last(self):
        self.assertEqual(resolve_attack_tasks("2,last", 6), {2, 6})
        with self.assertRaises(ValueError):
            resolve_attack_tasks("7", 6)

    def test_freezing_batch_norm_stats_keeps_affine_parameters_trainable(self):
        batch_norm = nn.BatchNorm1d(2)
        block = nn.Sequential(nn.Linear(2, 2), batch_norm)
        block.train()
        self.assertEqual(freeze_batch_norm_running_stats((block,)), 1)
        self.assertFalse(batch_norm.training)
        self.assertTrue(batch_norm.weight.requires_grad)
        block(torch.ones(3, 2)).sum().backward()
        self.assertIsNotNone(batch_norm.weight.grad)

    def test_pseudo_label_loss_has_no_training_label_argument(self):
        self.assertEqual(
            list(inspect.signature(hard_pseudo_label_loss).parameters),
            ["student_logits", "guiding_logits"],
        )
        student = torch.tensor([[2.0, -1.0], [-2.0, 3.0]], requires_grad=True)
        guide = torch.tensor([[4.0, 0.0], [1.0, 5.0]])
        loss, pseudo = hard_pseudo_label_loss(student, guide)
        self.assertEqual(pseudo.tolist(), [0, 1])
        self.assertGreater(float(loss.detach()), 0.0)

    def test_ewc_penalty_is_zero_at_anchor_and_positive_after_change(self):
        blocks = tiny_blocks()
        parameters = named_trainable_parameters(blocks)
        strategy = QuadraticImportanceStrategy("ewc", strength=2.0)
        importance = {
            name: torch.ones_like(parameter)
            for name, parameter in parameters
        }
        strategy.consolidate(parameters, importance)
        self.assertAlmostEqual(
            float(strategy.penalty(parameters).detach()), 0.0, places=7
        )
        with torch.no_grad():
            parameters[0][1].add_(0.5)
        self.assertAlmostEqual(
            float(strategy.penalty(parameters).detach()), 1.0, places=6
        )

    def test_taskwise_ewc_compression_preserves_weighted_center(self):
        blocks = tiny_blocks()
        parameters = named_trainable_parameters(blocks)
        name, parameter = parameters[0]
        strategy = QuadraticImportanceStrategy("ewc", strength=1.0)
        strategy.consolidate(parameters, {name: torch.ones_like(parameter)})
        with torch.no_grad():
            parameter.fill_(3.0)
        strategy.consolidate(parameters, {name: 3.0 * torch.ones_like(parameter)})
        expected = torch.tensor([[2.5, 2.0]])
        self.assertTrue(torch.allclose(strategy.anchor[name], expected))
        self.assertTrue(
            torch.allclose(strategy.importance[name], 4.0 * torch.ones_like(parameter))
        )

    def test_online_ewc_anchors_at_latest_parameters(self):
        blocks = tiny_blocks()
        parameters = named_trainable_parameters(blocks)
        name, parameter = parameters[0]
        strategy = QuadraticImportanceStrategy(
            "online_ewc",
            strength=1.0,
            decay=0.5,
        )
        strategy.consolidate(parameters, {name: torch.ones_like(parameter)})
        with torch.no_grad():
            parameter.fill_(2.0)
        strategy.consolidate(parameters, {name: 2.0 * torch.ones_like(parameter)})
        self.assertTrue(torch.allclose(strategy.anchor[name], parameter.detach()))
        self.assertTrue(
            torch.allclose(strategy.importance[name], 2.5 * torch.ones_like(parameter))
        )

    def test_quadratic_strategy_state_and_curvature_can_be_rolled_back(self):
        blocks = tiny_blocks()
        parameters = named_trainable_parameters(blocks)
        name, parameter = parameters[0]
        strategy = QuadraticImportanceStrategy("online_ewc", strength=3.0)
        strategy.consolidate(parameters, {name: 2.0 * torch.ones_like(parameter)})
        snapshot = strategy.state_dict()
        expected_anchor = strategy.anchor[name].clone()
        expected_curvature = 12.0 * torch.ones_like(parameter)
        self.assertTrue(
            torch.allclose(strategy.curvature(parameters)[name], expected_curvature)
        )

        with torch.no_grad():
            parameter.add_(4.0)
        strategy.consolidate(parameters, {name: 5.0 * torch.ones_like(parameter)})
        strategy.load_state_dict(snapshot, parameters)
        self.assertTrue(torch.allclose(strategy.anchor[name], expected_anchor))
        self.assertTrue(
            torch.allclose(strategy.curvature(parameters)[name], expected_curvature)
        )

    def test_synaptic_intelligence_accumulates_positive_importance(self):
        blocks = tiny_blocks()
        parameters = named_trainable_parameters(blocks)
        name, parameter = parameters[0]
        strategy = SynapticIntelligenceStrategy(strength=1.0, xi=0.1)
        strategy.begin_task(parameters)
        parameter.grad = torch.ones_like(parameter)
        strategy.capture_step(parameters)
        with torch.no_grad():
            parameter.add_(-0.1)
        strategy.finish_step(parameters)
        strategy.consolidate(parameters)
        self.assertTrue((strategy.importance[name] > 0).all())
        self.assertAlmostEqual(
            float(strategy.penalty(parameters).detach()), 0.0, places=7
        )


if __name__ == "__main__":
    unittest.main()
