#!/usr/bin/env python3
"""Post-hoc fixed-budget plasticity and spectrum probe for real BrainUICL checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def parse_int_list(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("integer list must not be empty")
    return result


def load_checkpoint(args, checkpoint_dir: Path):
    blocks = build_blocks(args)
    for name, block in zip(CHECKPOINT_NAMES, blocks):
        path = checkpoint_dir / f"{name}_parameter_{args.seed}.pkl"
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


def effective_rank(values: torch.Tensor, eps: float = 1e-12) -> dict[str, float | int]:
    matrix = values.detach().float().reshape(-1, values.shape[-1])
    if matrix.shape[0] > 1:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    total = singular.sum()
    if float(total.item()) <= eps:
        return {"effective_rank": 0.0, "stable_rank": 0.0, "rank90": 0, "rank95": 0, "sigma_max": 0.0}
    probability = singular / total
    entropy = -(probability * probability.clamp_min(eps).log()).sum()
    squared = singular.square()
    energy = torch.cumsum(squared / squared.sum().clamp_min(eps), dim=0)
    stable = squared.sum() / squared.max().clamp_min(eps)

    def energy_rank_at(threshold: float) -> int:
        target = torch.tensor(threshold, device=energy.device)
        return int(torch.searchsorted(energy, target).item() + 1)

    return {
        "effective_rank": float(entropy.exp().item()),
        "stable_rank": float(stable.item()),
        "rank90": energy_rank_at(0.90),
        "rank95": energy_rank_at(0.95),
        "sigma_max": float(singular[0].item()),
    }


@torch.no_grad()
def monitor_spectrum(blocks, loader, args) -> dict:
    set_train(blocks, False)
    features = {"fusion": [], "transformer": [], "classifier_input": []}
    for batch_index, (eog, eeg, _labels) in enumerate(loader):
        if args.monitor_max_batches and batch_index >= args.monitor_max_batches:
            break
        eog = eog.to(args.device).reshape(-1, args.model_param.EogNum, args.model_param.EpochLength)
        eeg = eeg.to(args.device).reshape(-1, args.model_param.EegNum, args.model_param.EpochLength)
        fused = blocks[0](eeg, eog)
        encoded = blocks[1](fused)
        classifier_input = blocks[2].sleep_stage_mlp(encoded)
        features["fusion"].append(fused.detach())
        features["transformer"].append(encoded.detach())
        features["classifier_input"].append(classifier_input.detach())
    if not all(features.values()):
        raise RuntimeError("monitor loader produced no batches")
    return {name: effective_rank(torch.cat(values, dim=0)) for name, values in features.items()}


def weight_norms(blocks) -> dict[str, float]:
    result = {}
    total = 0.0
    for name, block in zip(CHECKPOINT_NAMES, blocks):
        squared = sum(float(parameter.detach().float().square().sum().item()) for parameter in block.parameters())
        result[f"{name}_l2"] = math.sqrt(squared)
        total += squared
    result["global_l2"] = math.sqrt(total)
    return result


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
    parser.add_argument("--dataset", choices=("ISRUC",), default="ISRUC")
    parser.add_argument("--data-root", type=Path, default=Path("/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32"))
    parser.add_argument("--input-checkpoint-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/model_parameter"))
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "experiments" / "regularization_cl_eeg_runs" / "clean49_bn_frozen_e10_lr1e6_seed4321")
    parser.add_argument("--method", choices=("finetune", "ewc", "online_ewc", "si", "mas"), default="finetune")
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--stages", default="0,10,25")
    parser.add_argument("--probe-steps", default="0,10,20,50,100")
    parser.add_argument("--seed", type=int, default=4321)
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
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if not 0.0 < cli.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    stages = parse_int_list(cli.stages)
    probe_steps = parse_int_list(cli.probe_steps)
    if probe_steps[0] != 0:
        probe_steps = [0, *probe_steps]
    split_path = cli.split or cli.run_root / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    new_order = [int(value) for value in split["new_order"]]
    if any(stage < 0 or stage >= len(new_order) for stage in stages):
        parser.error(f"stages must be in 0..{len(new_order) - 1} so each stage has a next subject")
    device = torch.device(f"cuda:{cli.gpu}" if cli.gpu >= 0 and torch.cuda.is_available() else "cpu")
    args = SimpleNamespace(**vars(cli))
    args.device = device
    args.model_param = ModelConfig(cli.dataset)
    args.probe_steps = probe_steps
    fix_randomness(cli.seed)

    tasks = []
    for stage in stages:
        subject = new_order[stage]
        blocks = load_checkpoint(args, checkpoint_dir(args, stage))
        train_loader, eval_loader, train_sequences, eval_sequences = split_subject(args, subject)
        spectra = monitor_spectrum(blocks, eval_loader, args)
        norms = weight_norms(blocks)
        plasticity = fixed_budget_probe(blocks, train_loader, eval_loader, args)
        tasks.append(
            {
                "task": stage,
                "stage": stage,
                "subject": subject,
                "checkpoint_dir": str(checkpoint_dir(args, stage)),
                "train_sequences": train_sequences,
                "eval_sequences": eval_sequences,
                "plasticity": plasticity,
                "spectra": spectra,
                "weight_norms": norms,
            }
        )
        del blocks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rank = np.asarray([task["spectra"]["transformer"]["effective_rank"] for task in tasks], dtype=np.float64)
    gains = np.asarray([task["plasticity"]["acc_gain"] for task in tasks], dtype=np.float64)
    correlation = float(np.corrcoef(rank, gains)[0, 1]) if len(tasks) >= 2 and rank.std() > 0 and gains.std() > 0 else None
    result = {
        "protocol": "raeeg-lop-posthoc-v1",
        "config": {
            "dataset": cli.dataset,
            "method": cli.method,
            "seed": cli.seed,
            "stages": stages,
            "probe_steps": probe_steps,
            "probe_protocol": "supervised-oracle-fixed-budget-heldout",
            "train_fraction": cli.train_fraction,
            "max_sequences": cli.max_sequences,
            "device": str(device),
        },
        "tasks": tasks,
        "summary": {
            "mean_plasticity_acc_gain": float(gains.mean()),
            "mean_transformer_effective_rank": float(rank.mean()),
            "transformer_effective_rank_vs_future_plasticity_pearson": correlation,
            "stage_count": len(tasks),
        },
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
