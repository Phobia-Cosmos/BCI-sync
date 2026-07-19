import inspect
import unittest

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
    brainwash_one_step_batch,
    materialize_poisoned_subject,
    pacol_gradient_matching_batch,
)
from experiments.regularization_cl_eeg import resolve_attack_tasks


def tiny_blocks():
    first = nn.Linear(2, 1, bias=False)
    second = nn.Identity()
    third = nn.Identity()
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, -1.0]]))
    return first, second, third


class RegularizationCLEEGTest(unittest.TestCase):
    def test_attack_interfaces_do_not_accept_true_labels(self):
        for function in (
            pacol_gradient_matching_batch,
            brainwash_one_step_batch,
            materialize_poisoned_subject,
        ):
            parameter_names = set(inspect.signature(function).parameters)
            self.assertNotIn("labels", parameter_names)
            self.assertNotIn("true_labels", parameter_names)

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
