#!/usr/bin/env python3
"""Deployable unlabeled SPR-EEG with guiding-model pseudo-labels."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rttdp_brainuicl_full import (  # noqa: E402
    CPCProbe,
    SequenceDataset,
    clone_blocks,
    discover_subjects,
    evaluate,
    forward_blocks,
    load_pretrained,
    make_loader,
    set_train,
    split_subjects,
    subject_paths,
)
from experiments.spr_eeg_pure import (  # noqa: E402
    EEGSimCLR,
    buffer_stats,
    expert_embeddings,
    finetune_for_evaluation,
    plasticity_summary,
    source_records,
    train_base,
    train_expert,
)
from experiments.spr_eeg_unlabeled.filtering import filter_pseudo_labeled_epochs  # noqa: E402
from model.spr_eeg import PurifiedEpochBuffer, symmetric_label_noise  # noqa: E402
from utils.config import ModelConfig  # noqa: E402
from utils.util import compute_aaf1, compute_aaa, compute_forget, fix_randomness  # noqa: E402


def make_subject_loader(paths, args, shuffle):
    return DataLoader(
        SequenceDataset(paths),
        batch_size=args.batch if shuffle else args.eval_batch,
        shuffle=shuffle,
        num_workers=args.num_worker,
        drop_last=False,
    )


def train_guiding_model(blocks, loader, args):
    set_train(blocks, True)
    probe = CPCProbe(blocks, args)
    losses = []
    for _ in range(args.guiding_epochs):
        for batch_index, (eog, eeg, _labels) in enumerate(loader):
            if args.max_guiding_batches and batch_index >= args.max_guiding_batches:
                break
            eog, eeg = eog.to(args.device), eeg.to(args.device)
            loss, blocks = probe.update(eeg, eog)
            losses.append(loss)
    return blocks, float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def generate_all_pseudo_labels(blocks, loader, args):
    """Predict every epoch; labels from the loader are diagnostics only."""

    set_train(blocks, False)
    pseudo_labels, confidences, true_labels = [], [], []
    for eog, eeg, labels in loader:
        eog, eeg = eog.to(args.device), eeg.to(args.device)
        probabilities = forward_blocks(blocks, eog, eeg, args).softmax(dim=1)
        confidence, pseudo = probabilities.max(dim=1)
        pseudo_labels.append(pseudo.cpu().numpy())
        confidences.append(confidence.cpu().numpy())
        true_labels.append(labels.numpy())
    return (
        np.concatenate(pseudo_labels, axis=0),
        np.concatenate(confidences, axis=0),
        np.concatenate(true_labels, axis=0),
    )


def append_stability(performance, result):
    stability = performance["stability"]
    stability["ACC"].append(result["acc"])
    stability["MF1"].append(result["mf1"])
    stability["AAA"].append(float(compute_aaa(stability["ACC"])))
    stability["AAF1"] = compute_aaf1(stability["MF1"])
    stability["FR"].append(float(compute_forget(stability["ACC"])))


def save_progress(path, performance, config):
    payload = {
        "config": config,
        "performance": performance,
        "summary": plasticity_summary(performance),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def run(args):
    fix_randomness(args.seed)
    subjects = discover_subjects(args.data_root)
    train_idx, val_idx, old_idx, new_idx = split_subjects(subjects, args.seed)
    new_order = [int(subject) for subject in new_idx]
    if args.max_subjects > 0:
        new_order = new_order[: args.max_subjects]

    initial_blocks = load_pretrained(args)
    base = EEGSimCLR(
        copy.deepcopy(initial_blocks[0]).to(args.device),
        copy.deepcopy(initial_blocks[1]).to(args.device),
    ).to(args.device)
    base_optimizer = torch.optim.Adam(base.parameters(), lr=args.ssl_lr, weight_decay=args.weight_decay)
    memory = PurifiedEpochBuffer(
        args.memory_capacity,
        source_capacity=args.source_capacity,
        num_classes=args.model_param.NumClasses,
    )
    memory.seed_source(source_records(args.data_root, train_idx), seed=args.seed)
    inference_blocks = clone_blocks(initial_blocks, args)
    old_loader = make_loader(args.data_root, sorted(old_idx), args.eval_batch, False, args.num_worker)

    config = {
        "method": "unlabeled_spr_eeg",
        "seed": args.seed,
        "guiding_model": "CPC-adapted previous inference model",
        "new_subject_ground_truth_used_for_training": False,
        "confidence_gate": False,
        "candidate_rule": "all pseudo-labeled epochs",
        "expert_epochs": args.expert_epochs,
        "base_epochs": args.base_epochs,
        "guiding_epochs": args.guiding_epochs,
        "ft_epochs": args.ft_epochs,
        "memory_capacity": args.memory_capacity,
        "source_capacity": args.source_capacity,
        "extra_pseudo_noise": args.extra_pseudo_noise,
        "split": {
            "train": sorted(int(subject) for subject in train_idx),
            "val": sorted(int(subject) for subject in val_idx),
            "old": sorted(int(subject) for subject in old_idx),
            "new_order": new_order,
        },
    }
    performance = {
        "completed_subjects": 0,
        "stability": {"ACC": [], "MF1": [], "AAA": [], "AAF1": [], "FR": []},
        "plasticity": {str(subject): {"ACC": [], "MF1": []} for subject in new_order},
        "steps": [],
    }
    append_stability(performance, evaluate(inference_blocks, old_loader, args))

    for step, subject in enumerate(new_order, start=1):
        paths = subject_paths(args.data_root, subject)
        train_loader = make_subject_loader(paths, args, True)
        ordered_loader = make_subject_loader(paths, args, False)
        previous_blocks = clone_blocks(inference_blocks, args)
        initial_result = evaluate(initial_blocks, ordered_loader, args)
        before_result = evaluate(previous_blocks, ordered_loader, args)

        guiding_blocks = clone_blocks(previous_blocks, args)
        guiding_blocks, guiding_loss = train_guiding_model(guiding_blocks, train_loader, args)
        pseudo_labels, confidences, diagnostic_true_labels = generate_all_pseudo_labels(
            guiding_blocks, ordered_loader, args
        )
        if args.extra_pseudo_noise > 0:
            pseudo_labels, extra_noise_mask = symmetric_label_noise(
                pseudo_labels,
                args.extra_pseudo_noise,
                args.model_param.NumClasses,
                seed=args.seed + 3000 * step,
            )
        else:
            extra_noise_mask = np.zeros_like(pseudo_labels, dtype=bool)

        expert = EEGSimCLR(
            copy.deepcopy(initial_blocks[0]).to(args.device),
            copy.deepcopy(initial_blocks[1]).to(args.device),
        ).to(args.device)
        expert_loss = train_expert(expert, train_loader, args)
        base_loss = train_base(base, base_optimizer, train_loader, memory, args)
        features = expert_embeddings(expert, ordered_loader, args)
        filtered = filter_pseudo_labeled_epochs(
            features,
            pseudo_labels,
            paths[0],
            ensembles=args.ensembles,
            bmm_iters=args.bmm_iters,
            seed=args.seed + 10000 * step,
            true_labels_for_diagnostics=diagnostic_true_labels,
        )
        memory.update(filtered.records)
        inference_blocks, ft_loss = finetune_for_evaluation(
            base, initial_blocks[2], memory, args
        )
        after_result = evaluate(inference_blocks, ordered_loader, args)
        old_result = evaluate(inference_blocks, old_loader, args)
        append_stability(performance, old_result)
        performance["plasticity"][str(subject)] = {
            "ACC": [initial_result["acc"], before_result["acc"], after_result["acc"]],
            "MF1": [initial_result["mf1"], before_result["mf1"], after_result["mf1"]],
        }
        filter_metrics = dict(filtered.metrics)
        filter_metrics.update(
            {
                "mean_guiding_confidence": float(confidences.mean()),
                "epochs_below_0_9": int((confidences < 0.9).sum()),
                "extra_noisy_epochs": int(extra_noise_mask.sum()),
            }
        )
        step_row = {
            "step": step,
            "subject": subject,
            "guiding_cpc_loss": guiding_loss,
            "expert_nt_xent": expert_loss,
            "base_nt_xent": base_loss,
            "finetune_loss": ft_loss,
            "filter": filter_metrics,
            "buffer": buffer_stats(memory),
            "old_acc": old_result["acc"],
            "old_mf1": old_result["mf1"],
            "new_initial_acc": initial_result["acc"],
            "new_before_acc": before_result["acc"],
            "new_after_acc": after_result["acc"],
            "new_after_mf1": after_result["mf1"],
        }
        performance["steps"].append(step_row)
        performance["completed_subjects"] = step
        save_progress(args.output_root / "metrics.json", performance, config)
        print(
            f"[unlabeled SPR] {step}/{len(new_order)} subject={subject} "
            f"old={old_result['acc']:.4f} new before/after={before_result['acc']:.4f}/{after_result['acc']:.4f} "
            f"pseudo_err={filter_metrics['pseudo_label_error_before']:.4f}->"
            f"{filter_metrics['pseudo_label_error_after']:.4f} P={len(memory)}",
            flush=True,
        )

    save_progress(args.output_root / "metrics.json", performance, config)
    return performance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"))
    parser.add_argument("--input-checkpoint-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/model_parameter"))
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "rttdp_brainuicl_runs" / "spr_unlabeled")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=10)
    parser.add_argument("--guiding-epochs", type=int, default=10)
    parser.add_argument("--expert-epochs", type=int, default=10)
    parser.add_argument("--base-epochs", type=int, default=10)
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--max-guiding-batches", type=int, default=0)
    parser.add_argument("--max-ssl-batches", type=int, default=0)
    parser.add_argument("--max-ft-batches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssl-lr", type=float, default=1e-6)
    parser.add_argument("--ft-lr", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--jitter", type=float, default=0.01)
    parser.add_argument("--mask-ratio", type=float, default=0.1)
    parser.add_argument("--channel-drop", type=float, default=0.1)
    parser.add_argument("--ensembles", type=int, default=5)
    parser.add_argument("--bmm-iters", type=int, default=10)
    parser.add_argument("--memory-capacity", type=int, default=5000)
    parser.add_argument("--source-capacity", type=int, default=3000)
    parser.add_argument("--extra-pseudo-noise", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    fix_randomness(args.seed)
    args.dataset = "ISRUC"
    args.model_param = ModelConfig(args.dataset)
    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()

