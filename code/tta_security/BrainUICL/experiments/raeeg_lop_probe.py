#!/usr/bin/env python3
"""Post-hoc LoP diagnostics for real BrainUICL checkpoints.

The v2 entry point keeps the native BrainUICL trainer untouched while using
the shared EdgeForge metrics for layer spectra, CKA/Procrustes drift,
Jacobian/NTK, gradients, attention, and a fixed-budget fresh-vs-warm probe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the architecture-agnostic metric implementation from EdgeForge.  An
# explicit override keeps this integration portable while the default matches
# the shared workspace layout.
EDGEFORGE_SRC = Path(
    os.environ.get("EDGEFORGE_SRC", "/home/undefined/Desktop/EdgeForge/src")
)
if EDGEFORGE_SRC.is_dir() and str(EDGEFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(EDGEFORGE_SRC))

from edgeforge.lop_metrics import (  # noqa: E402
    activation_summary,
    attention_summary,
    fixed_budget_probe as generic_fixed_budget_probe,
    gradient_summary,
    jacobian_summary,
    linear_cka,
    local_linearity_summary,
    parameter_norm_summary,
    procrustes_residual,
    sampled_parameter_jacobian,
    spectral_summary,
)

from model.regularization_cl import freeze_batch_norm_running_stats  # noqa: E402
from rttdp_brainuicl_full import (  # noqa: E402
    SequenceDataset,
    build_blocks,
    clone_blocks,
    evaluate,
    flat_labels,
    flat_logits,
    forward_blocks,
    set_train,
    subject_paths,
)
from utils.config import ModelConfig  # noqa: E402
from utils.util import fix_randomness  # noqa: E402


CHECKPOINT_NAMES = ("feature_extractor", "feature_encoder", "sleep_classifier")


class BrainUICLModel(nn.Module):
    """Single-module view of BrainUICL for the shared LoP metric API.

    BrainUICL stores the frontend, temporal encoder and classifier as a tuple
    and receives EOG/EEG separately.  The wrapper packs both modalities into
    ``[B, T, C, S]`` so generic Jacobian and fresh-vs-warm probes can operate
    on the real model without modifying its training code.
    """

    def __init__(self, blocks, args):
        super().__init__()
        self.feature_extractor, self.feature_encoder, self.classifier = blocks
        self.args = args
        self.model_param = args.model_param

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 4:
            raise ValueError(
                f"packed BrainUICL input must be [B,T,C,S], got {tuple(packed.shape)}"
            )
        expected_channels = self.model_param.EogNum + self.model_param.EegNum
        if packed.shape[2] != expected_channels:
            raise ValueError(
                f"packed input has {packed.shape[2]} channels; expected {expected_channels}"
            )
        eog = packed[:, :, : self.model_param.EogNum, :]
        eeg = packed[:, :, self.model_param.EogNum :, :]
        # The shared evaluator expects classes in the final dimension, while
        # the native BrainUICL classifier returns [B, classes, T].
        return forward_blocks(
            (self.feature_extractor, self.feature_encoder, self.classifier),
            eog,
            eeg,
            self.args,
        ).permute(0, 2, 1).contiguous()


def pack_loader_batches(loader, *, max_batches: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialize ``(packed_input, labels)`` batches for generic probes."""

    batches = []
    for batch_index, (eog, eeg, labels) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batches.append((torch.cat((eog, eeg), dim=2).contiguous(), labels.contiguous()))
    if not batches:
        raise RuntimeError("loader produced no batches for LoP diagnostics")
    return batches


def parse_int_list(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("integer list must not be empty")
    return result


def load_checkpoint(args, checkpoint_dir: Path, *, checkpoint_seed: int | None = None):
    blocks = build_blocks(args)
    seed = args.seed if checkpoint_seed is None else int(checkpoint_seed)
    for name, block in zip(CHECKPOINT_NAMES, blocks):
        path = checkpoint_dir / f"{name}_parameter_{seed}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        block.load_state_dict(torch.load(path, map_location=args.device))
    return blocks


def split_subject(args, subject: int):
    data_paths, label_paths = subject_paths(args.data_root, subject)
    indices = np.arange(len(data_paths))
    np.random.default_rng(args.seed + subject).shuffle(indices)
    if args.max_sequences > 0:
        indices = indices[: args.max_sequences]
    if len(indices) < 2:
        raise RuntimeError(f"subject {subject} needs at least two sequences for a held-out probe")
    split = min(len(indices) - 1, max(1, int(round(len(indices) * args.train_fraction))))
    train_indices = indices[:split]
    eval_indices = indices[split:]

    def paths(selected):
        return ([data_paths[int(i)] for i in selected], [label_paths[int(i)] for i in selected])

    train_loader = DataLoader(
        SequenceDataset(paths(train_indices)),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_worker,
    )
    eval_loader = DataLoader(
        SequenceDataset(paths(eval_indices)),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    return train_loader, eval_loader, len(train_indices), len(eval_indices)


def retention_loader(args, subject: int) -> DataLoader:
    """Build a deterministic old-subject evaluation loader."""

    paths = subject_paths(args.data_root, subject)
    if not paths[0]:
        raise RuntimeError(f"subject {subject} has no processed sequence files under {args.data_root}")
    return DataLoader(
        SequenceDataset(paths),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_worker,
    )


def effective_rank(values: torch.Tensor, eps: float = 1e-12) -> dict[str, float | int]:
    summary = spectral_summary(values, feature_axis=-1, epsilon=max(float(eps), 1e-6))
    return {
        key: summary[key]
        for key in (
            "effective_rank",
            "effective_rank_normalized",
            "stable_rank",
            "stable_rank_normalized",
            "rank90",
            "rank95",
            "rank99",
            "sigma_max",
            "sigma_min_positive",
            "condition_number",
            "tail_energy",
            "observation_count",
            "feature_dim",
        )
    }


@torch.no_grad()
def collect_representations(blocks, loader, args, *, max_batches: int = 0) -> dict[str, torch.Tensor]:
    """Collect fixed-data representations without retaining autograd graphs."""

    set_train(blocks, False)
    features = {"fusion": [], "transformer": [], "classifier_input": [], "logits": [], "attention": []}
    for batch_index, (eog, eeg, _labels) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        eog = eog.to(args.device).reshape(-1, args.model_param.EogNum, args.model_param.EpochLength)
        eeg = eeg.to(args.device).reshape(-1, args.model_param.EegNum, args.model_param.EpochLength)
        fused = blocks[0](eeg, eog)
        encoded = blocks[1](fused)
        classifier_input = blocks[2].sleep_stage_mlp(encoded)
        logits = blocks[2](encoded)
        features["fusion"].append(fused.detach().cpu())
        features["transformer"].append(encoded.detach().cpu())
        features["classifier_input"].append(classifier_input.detach().cpu())
        # SleepMLP returns [B, classes, T]; expose logits as [B, T, classes]
        # so the feature axis is always the final axis in metric summaries.
        features["logits"].append(logits.permute(0, 2, 1).detach().cpu())
        attention = getattr(getattr(blocks[1], "encoder", None), "multi_attention", None)
        attention = getattr(attention, "last_attention_prob", None)
        if attention is not None:
            features["attention"].append(attention.detach().cpu())
    if not features["fusion"]:
        raise RuntimeError("monitor loader produced no batches")
    result = {
        name: torch.cat(values, dim=0)
        for name, values in features.items()
        if values
    }
    return result


def representation_diagnostics(
    representations: dict[str, torch.Tensor],
    *,
    previous: dict[str, torch.Tensor] | None = None,
    max_observations: int = 0,
) -> dict[str, dict]:
    """Compute layer-wise spectra, activation, and rotation-aware drift."""

    kinds = {"fusion": "gelu", "transformer": "unknown", "classifier_input": "gelu", "logits": "logit"}
    result = {}
    for name in ("fusion", "transformer", "classifier_input", "logits"):
        values = representations.get(name)
        if values is None:
            continue
        item = {
            "representation_level": "epoch_token" if name != "logits" else "epoch_logits",
            "feature_axis": -1,
            "spectrum": spectral_summary(values, feature_axis=-1, max_observations=max_observations),
            "activation": activation_summary(values, kind=kinds[name]),
        }
        if previous is not None and previous.get(name) is not None:
            item["drift"] = {
                "cka": linear_cka(values, previous[name], feature_axis=-1, max_observations=max_observations),
                "procrustes_residual": procrustes_residual(
                    values,
                    previous[name],
                    feature_axis=-1,
                    max_observations=max_observations,
                ),
            }
        result[name] = item
    return result


def monitor_spectrum(blocks, loader, args) -> dict:
    """Backward-compatible spectrum view with the richer shared metrics."""

    representations = collect_representations(
        blocks,
        loader,
        args,
        max_batches=getattr(args, "monitor_max_batches", 0),
    )
    return {
        name: spectral_summary(
            values,
            feature_axis=-1,
            max_observations=getattr(args, "max_observations", 0),
        )
        for name, values in representations.items()
        if name != "attention"
    }


def weight_norms(blocks) -> dict[str, float]:
    result = {}
    total = 0.0
    for name, block in zip(CHECKPOINT_NAMES, blocks):
        squared = sum(float(parameter.detach().float().square().sum().item()) for parameter in block.parameters())
        result[f"{name}_l2"] = math.sqrt(squared)
        total += squared
    result["global_l2"] = math.sqrt(total)
    return result


def _collect_gradient_matrix(model: nn.Module, batches, args) -> torch.Tensor:
    """Collect per-batch supervised-oracle gradients for the metric summary."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    rows = []
    model.eval()
    for packed, labels in batches:
        packed = packed.to(args.device)
        labels = labels.to(args.device)
        model.zero_grad(set_to_none=True)
        logits = model(packed)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1).long())
        loss.backward()
        rows.append(
            torch.cat(
                [
                    (
                        parameter.grad
                        if parameter.grad is not None
                        else torch.zeros_like(parameter)
                    ).detach().float().reshape(-1)
                    for parameter in parameters
                ]
            ).cpu()
        )
    model.zero_grad(set_to_none=True)
    return torch.stack(rows) if rows else torch.empty((0, sum(int(p.numel()) for p in parameters)))


def model_lop_diagnostics(
    blocks,
    loader,
    args,
    *,
    previous_representations: dict[str, torch.Tensor] | None = None,
    seed: int,
    include_expensive: bool = True,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run the shared LoP metrics on a real BrainUICL checkpoint.

    The labels used here are read only for the supervised-oracle gradient and
    Jacobian probes.  They are never used by the continual-learning update.
    """

    representations = collect_representations(
        blocks,
        loader,
        args,
        max_batches=getattr(args, "diagnostic_max_batches", 4),
    )
    diagnostics: dict[str, Any] = {
        "protocol": "edgeforge-lop-metrics-v1-brainuicl",
        "label_source": "true_eval_labels_for_oracle_diagnostics",
        "layers": representation_diagnostics(
            representations,
            previous=previous_representations,
            max_observations=getattr(args, "max_observations", 0),
        ),
        "attention": {"status": "unavailable", "reason": "BrainUICL attention capture is not exposed"},
        "parameters": parameter_norm_summary(BrainUICLModel(blocks, args)),
    }
    attention = representations.get("attention")
    if attention is not None:
        # BrainUICL normalizes across heads (dim=1), unlike standard
        # key-axis attention.  Do not present diagonal mass as key locality.
        attention_item = attention_summary(attention, normalization_axis=1)
        attention_item.pop("diagonal_mass", None)
        attention_item.pop("offdiagonal_mass", None)
        attention_item["axis_semantics"] = "head-normalized; key-axis locality unavailable"
        diagnostics["attention"] = attention_item

    if include_expensive:
        packed_batches = pack_loader_batches(
            loader,
            max_batches=getattr(args, "diagnostic_max_batches", 0),
        )
        model = BrainUICLModel(blocks, args).to(args.device)
        first_inputs = packed_batches[0][0].to(args.device)
        jacobian = sampled_parameter_jacobian(
            model,
            first_inputs,
            max_samples=max(1, int(getattr(args, "jacobian_samples", 2))),
        )
        diagnostics["jacobian"] = jacobian_summary(jacobian)
        gradients = _collect_gradient_matrix(model, packed_batches, args)
        diagnostics["gradient"] = gradient_summary(gradients)
        diagnostics["local_linearity"] = local_linearity_summary(
            model,
            first_inputs,
            epsilons=(float(getattr(args, "linearity_epsilon", 1e-3)),),
            directions=max(1, int(getattr(args, "linearity_directions", 2))),
            seed=int(seed),
        )
        diagnostics["parameters"] = parameter_norm_summary(model)
    return diagnostics, representations


def fresh_warm_probe(
    warm_blocks,
    fresh_blocks,
    train_loader,
    eval_loader,
    args,
    *,
    seed: int,
) -> dict[str, Any]:
    """Compare a continual checkpoint with a source/fresh reference."""

    train_batches = pack_loader_batches(
        train_loader,
        max_batches=max(1, int(getattr(args, "probe_train_batches", 4))),
    )
    eval_batches = pack_loader_batches(
        eval_loader,
        max_batches=max(1, int(getattr(args, "probe_eval_batches", 4))),
    )
    result = generic_fixed_budget_probe(
        BrainUICLModel(warm_blocks, args),
        BrainUICLModel(fresh_blocks, args),
        train_batches,
        eval_batches,
        steps=args.probe_steps,
        lr=args.probe_lr,
        weight_decay=args.weight_decay,
        seed=seed,
        freeze_batch_norm=bool(getattr(args, "freeze_bn_stats", False)),
    )
    result.update(
        {
            "label_source": "true_labels_for_fixed_budget_oracle_probe",
            "train_batch_count": len(train_batches),
            "eval_batch_count": len(eval_batches),
            "outcome_definition": {
                "fresh_gap_final": "accuracy_fresh - accuracy_warm",
                "fresh_loss_gap_final": "loss_warm - loss_fresh",
                "fresh_auc_gap": "AULC_fresh - AULC_warm",
            },
        }
    )
    return result


def evaluate_retention(blocks, subjects: list[int], args) -> dict[str, Any]:
    """Evaluate fixed old subjects without touching model parameters."""

    rows = []
    for subject in subjects:
        loader = retention_loader(args, int(subject))
        metrics = evaluate(
            blocks,
            loader,
            args,
            max_batches=int(getattr(args, "retention_max_batches", 0)),
        )
        rows.append(
            {
                "subject": int(subject),
                "accuracy": float(metrics["acc"]),
                "macro_f1": float(metrics["mf1"]),
                "count": int(metrics["n_epochs"]),
            }
        )
    return {
        "status": "computed" if rows else "not-requested",
        "label_source": "true_labels_for_retention_eval_only",
        "subjects": rows,
        "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])) if rows else None,
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])) if rows else None,
    }


def write_markdown_report(result: dict[str, Any], path: Path) -> None:
    """Write a compact human-readable companion to the machine JSON."""

    config = result["config"]
    summary = result["summary"]
    lines = [
        "# BrainUICL LoP diagnostics",
        "",
        f"- Protocol: `{result['protocol']}`",
        f"- Dataset/method: `{config['dataset']}` / `{config['method']}`",
        f"- Stages: `{config['stages']}`; seed: `{config['seed']}`",
        f"- Fresh comparator: `{config['fresh_reference']}`",
        f"- Primary outcome: `{result['primary_outcome']}`",
        "- Labels are used only by the held-out/oracle diagnostics; the native unsupervised trainer is unchanged.",
        f"- Retention subjects: `{config.get('retention_subjects', [])}` (eval-only)",
        "",
        "## Summary",
        "",
        f"- Mean warm accuracy gain: `{summary['mean_warm_acc_gain']:+.6f}`",
        f"- Mean final fresh gap: `{summary['mean_fresh_gap_final']:+.6f}`",
        f"- Mean fresh AULC gap: `{summary['mean_fresh_auc_gap']:+.6f}`",
        f"- Mean Transformer effective rank: `{summary['mean_transformer_effective_rank']:.6f}`",
        f"- Rank/fresh-gap Pearson: `{summary['transformer_effective_rank_vs_fresh_gap_pearson']}`",
        "",
        "## Stage Results",
        "",
        "| stage | target subject | Transformer ER | normalized ER | final fresh gap | fresh AULC gap | old ACC | old MF1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in result["tasks"]:
        spectrum = task["spectra"]["transformer"]
        outcome = task["plasticity"]["outcome"]
        retention = task.get("retention", {"mean_accuracy": None, "mean_macro_f1": None})
        old_acc = retention["mean_accuracy"]
        old_mf1 = retention["mean_macro_f1"]
        old_acc_text = "—" if old_acc is None else f"{old_acc:.6f}"
        old_mf1_text = "—" if old_mf1 is None else f"{old_mf1:.6f}"
        lines.append(
            f"| {task['stage']} | {task['subject']} | {spectrum['effective_rank']:.6f} | "
            f"{spectrum['effective_rank_normalized']:.6f} | {outcome['fresh_gap_final']:+.6f} | "
            f"{outcome['fresh_auc_gap']:+.6f} | {old_acc_text} | {old_mf1_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "A positive fresh gap is a LoP candidate signal only when the same target split, update budget, optimizer, and seed protocol are repeated across stages and at least three seeds. Rank, CKA/Procrustes, Jacobian/NTK, gradients, activations, and attention are mechanism diagnostics, not standalone LoP evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def fixed_budget_probe(blocks, train_loader, eval_loader, args) -> dict:
    probe = clone_blocks(blocks, args)
    optimizer = torch.optim.Adam(
        [parameter for block in probe for parameter in block.parameters() if parameter.requires_grad],
        lr=args.probe_lr,
        weight_decay=args.weight_decay,
    )
    milestones = args.probe_steps
    rows = []

    def record(step: int, loss: float | None = None) -> None:
        result = evaluate(probe, eval_loader, args, max_batches=args.eval_max_batches)
        rows.append(
            {
                "step": step,
                "acc": float(result["acc"]),
                "mf1": float(result["mf1"]),
                "loss": loss,
            }
        )

    record(0)
    iterator = iter(train_loader)
    for step in range(1, milestones[-1] + 1):
        try:
            eog, eeg, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            eog, eeg, labels = next(iterator)
        set_train(probe, True)
        if args.freeze_bn_stats:
            freeze_batch_norm_running_stats(probe)
        eog = eog.to(args.device)
        eeg = eeg.to(args.device)
        labels = labels.to(args.device)
        logits = forward_blocks(probe, eog, eeg, args)
        loss = F.cross_entropy(flat_logits(logits), flat_labels(labels))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in milestones:
            record(step, float(loss.detach().item()))
    steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
    accuracies = np.asarray([row["acc"] for row in rows], dtype=np.float64)
    aulc = float(np.trapezoid(accuracies, steps) / steps[-1]) if steps[-1] > 0 else float(accuracies[0])
    return {
        "protocol": "supervised-oracle-fixed-budget-heldout",
        "curve": rows,
        "acc_gain": float(rows[-1]["acc"] - rows[0]["acc"]),
        "mf1_gain": float(rows[-1]["mf1"] - rows[0]["mf1"]),
        "aulc": aulc,
    }


def checkpoint_dir(args, stage: int) -> Path:
    if stage == 0:
        return args.input_checkpoint_root / args.dataset / "Pretrain"
    return args.run_root / args.method / "checkpoints" / f"individual_{stage}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ISRUC", "FACED"), default="ISRUC")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"),
    )
    parser.add_argument(
        "--input-checkpoint-root",
        type=Path,
        default=Path("/home/undefined/Disk/ai-storage/BrainUICL/model_parameter"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "regularization_cl_eeg_runs" / "clean49_bn_frozen_e10_lr1e6_seed4321",
    )
    parser.add_argument("--method", choices=("finetune", "ewc", "online_ewc", "si", "mas"), default="finetune")
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--stages", default="0,10,25")
    parser.add_argument("--probe-steps", default="0,10,20,50,100")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--pretrain-seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--probe-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--freeze-bn-stats", action="store_true")
    parser.add_argument("--monitor-max-batches", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument(
        "--fresh-reference",
        choices=("source", "random"),
        default="random",
        help="fresh comparator: strict random initialization (default) or source-pretrained baseline",
    )
    parser.add_argument(
        "--anchor-subject",
        type=int,
        default=-1,
        help="fixed subject for same-data CKA/Procrustes drift (-1 uses the first new subject)",
    )
    parser.add_argument(
        "--retention-subject",
        action="append",
        type=int,
        default=[],
        help="old subject to evaluate for retention/BWT; repeat the option for multiple subjects",
    )
    parser.add_argument(
        "--retention-max-batches",
        type=int,
        default=0,
        help="optional bound for each old-subject evaluation loader (0 = all)",
    )
    parser.add_argument(
        "--diagnostic-max-batches",
        type=int,
        default=4,
        help="bounded batches used for Jacobian/gradient and representation diagnostics",
    )
    parser.add_argument("--max-observations", type=int, default=256)
    parser.add_argument("--jacobian-samples", type=int, default=2)
    parser.add_argument("--linearity-epsilon", type=float, default=1e-3)
    parser.add_argument("--linearity-directions", type=int, default=2)
    parser.add_argument("--probe-train-batches", type=int, default=4)
    parser.add_argument("--probe-eval-batches", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if not 0.0 < cli.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    if cli.diagnostic_max_batches < 1 or cli.jacobian_samples < 1:
        parser.error("diagnostic-max-batches and jacobian-samples must be positive")
    if cli.retention_max_batches < 0:
        parser.error("retention-max-batches must be non-negative")
    stages = parse_int_list(cli.stages)
    probe_steps = parse_int_list(cli.probe_steps)
    if probe_steps[0] != 0:
        probe_steps = [0, *probe_steps]
    split_path = cli.split or cli.run_root / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    new_order = [int(value) for value in split["new_order"]]
    if any(stage < 0 or stage >= len(new_order) for stage in stages):
        parser.error(f"stages must be in 0..{len(new_order) - 1}")
    device = torch.device(f"cuda:{cli.gpu}" if cli.gpu >= 0 and torch.cuda.is_available() else "cpu")
    args = SimpleNamespace(**vars(cli))
    args.device = device
    args.model_param = ModelConfig(cli.dataset)
    args.probe_steps = probe_steps
    fix_randomness(cli.seed)

    anchor_subject = cli.anchor_subject if cli.anchor_subject >= 0 else new_order[0]
    _anchor_train_loader, anchor_eval_loader, _anchor_train_count, _anchor_eval_count = split_subject(args, anchor_subject)
    previous_anchor_representations = None
    source_dir = cli.input_checkpoint_root / cli.dataset / "Pretrain"
    tasks = []
    for stage in stages:
        subject = new_order[stage]
        current_dir = checkpoint_dir(args, stage)
        current_seed = cli.pretrain_seed if stage == 0 and cli.pretrain_seed is not None else cli.seed
        blocks = load_checkpoint(args, current_dir, checkpoint_seed=current_seed)
        train_loader, eval_loader, train_sequences, eval_sequences = split_subject(args, subject)

        target_diagnostics, _target_representations = model_lop_diagnostics(
            blocks,
            eval_loader,
            args,
            seed=cli.seed + 1000 + stage,
            include_expensive=True,
        )
        anchor_diagnostics, current_anchor_representations = model_lop_diagnostics(
            blocks,
            anchor_eval_loader,
            args,
            previous_representations=previous_anchor_representations,
            seed=cli.seed + 2000 + stage,
            include_expensive=False,
        )

        if cli.fresh_reference == "source":
            fresh_blocks = load_checkpoint(
                args,
                source_dir,
                checkpoint_seed=cli.pretrain_seed if cli.pretrain_seed is not None else cli.seed,
            )
        else:
            torch.manual_seed(cli.seed + 900000 + stage)
            fresh_blocks = build_blocks(args)
        plasticity = fresh_warm_probe(
            blocks,
            fresh_blocks,
            train_loader,
            eval_loader,
            args,
            seed=cli.seed + 3000 + stage,
        )
        retention = evaluate_retention(blocks, cli.retention_subject, args)
        spectra = {
            name: item["spectrum"]
            for name, item in target_diagnostics["layers"].items()
        }
        tasks.append(
            {
                "task": stage,
                "stage": stage,
                "subject": subject,
                "checkpoint_dir": str(current_dir),
                "train_sequences": train_sequences,
                "eval_sequences": eval_sequences,
                "plasticity": plasticity,
                "spectra": spectra,
                "weight_norms": weight_norms(blocks),
                "diagnostics": {
                    "target": target_diagnostics,
                    "anchor": anchor_diagnostics,
                },
                "retention": retention,
            }
        )
        previous_anchor_representations = current_anchor_representations
        del fresh_blocks, blocks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rank = np.asarray([task["spectra"]["transformer"]["effective_rank"] for task in tasks], dtype=np.float64)
    warm_gains = np.asarray([task["plasticity"]["outcome"]["warm_acc_gain"] for task in tasks], dtype=np.float64)
    fresh_gaps = np.asarray([task["plasticity"]["outcome"]["fresh_gap_final"] for task in tasks], dtype=np.float64)
    fresh_auc_gaps = np.asarray([task["plasticity"]["outcome"]["fresh_auc_gap"] for task in tasks], dtype=np.float64)
    correlation = float(np.corrcoef(rank, fresh_gaps)[0, 1]) if len(tasks) >= 2 and rank.std() > 0 and fresh_gaps.std() > 0 else None
    result = {
        "protocol": "raeeg-lop-posthoc-v2-edgeforge-shared",
        "primary_outcome": "fresh_gap_final = accuracy_fresh - accuracy_warm",
        "config": {
            "dataset": cli.dataset,
            "method": cli.method,
            "seed": cli.seed,
            "pretrain_seed": cli.pretrain_seed if cli.pretrain_seed is not None else cli.seed,
            "stages": stages,
            "probe_steps": probe_steps,
            "probe_protocol": "edgeforge-fixed-budget-fresh-vs-warm-heldout",
            "fresh_reference": cli.fresh_reference,
            "train_fraction": cli.train_fraction,
            "max_sequences": cli.max_sequences,
            "anchor_subject": anchor_subject,
            "retention_subjects": cli.retention_subject,
            "retention_max_batches": cli.retention_max_batches,
            "diagnostic_max_batches": cli.diagnostic_max_batches,
            "max_observations": cli.max_observations,
            "freeze_bn_stats": cli.freeze_bn_stats,
            "device": str(device),
        },
        "tasks": tasks,
        "summary": {
            "mean_warm_acc_gain": float(warm_gains.mean()),
            "mean_fresh_gap_final": float(fresh_gaps.mean()),
            "mean_fresh_auc_gap": float(fresh_auc_gaps.mean()),
            "mean_transformer_effective_rank": float(rank.mean()),
            "transformer_effective_rank_vs_fresh_gap_pearson": correlation,
            "stage_count": len(tasks),
        },
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(result, cli.output.with_suffix(".md"))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
