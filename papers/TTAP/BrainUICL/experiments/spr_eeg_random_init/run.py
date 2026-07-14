#!/usr/bin/env python3
"""Random-initialized SPR on EEG with progression and fixed-split evaluation."""

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
    SequenceDataset,
    build_blocks,
    clone_blocks,
    discover_subjects,
    evaluate,
    make_loader,
    split_subjects,
    subject_paths,
)
from experiments.spr_eeg_pure import (  # noqa: E402
    EEGSimCLR,
    buffer_stats,
    filter_delayed_buffer,
    finetune_for_evaluation,
    train_base,
    train_expert,
)
from experiments.spr_eeg_random_init.protocols import (  # noqa: E402
    PathPair,
    shuffled_subject_order,
    split_sequence_paths,
)
from model.spr_eeg import PurifiedEpochBuffer  # noqa: E402
from utils.config import ModelConfig  # noqa: E402
from utils.util import compute_aaf1, compute_aaa, compute_forget, fix_randomness  # noqa: E402


def make_path_loader(paths: PathPair, batch: int, shuffle: bool, workers: int):
    return DataLoader(
        SequenceDataset(paths),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
    )


def subject_train_test(args, subject: int) -> tuple[PathPair, PathPair]:
    return split_sequence_paths(
        subject_paths(args.data_root, subject),
        args.holdout_ratio,
        args.seed + 7919 * int(subject),
    )


def initialize_random_spr(args):
    initial_blocks = build_blocks(args)
    base = EEGSimCLR(
        copy.deepcopy(initial_blocks[0]).to(args.device),
        copy.deepcopy(initial_blocks[1]).to(args.device),
    ).to(args.device)
    optimizer = torch.optim.Adam(base.parameters(), lr=args.ssl_lr, weight_decay=args.weight_decay)
    memory = PurifiedEpochBuffer(args.memory_capacity, source_capacity=0, num_classes=args.model_param.NumClasses)
    return initial_blocks, base, optimizer, memory


def train_stream_chunk(args, base, base_optimizer, memory, paths, noise_rate, stream_step):
    current_loader = make_path_loader(paths, args.batch, True, args.num_worker)
    ordered_loader = make_path_loader(paths, args.eval_batch, False, args.num_worker)
    expert_blocks = build_blocks(args)
    expert = EEGSimCLR(expert_blocks[0], expert_blocks[1]).to(args.device)
    expert_loss = train_expert(expert, current_loader, args)
    base_loss = train_base(base, base_optimizer, current_loader, memory, args)
    accepted, filter_metrics = filter_delayed_buffer(
        expert, ordered_loader, paths, noise_rate, args, stream_step
    )
    memory.update(accepted)
    return {
        "expert_nt_xent": expert_loss,
        "base_nt_xent": base_loss,
        "filter": filter_metrics,
        "buffer": buffer_stats(memory),
    }


def make_inference(args, base, initial_classifier, memory):
    if not memory.records:
        return None, math.nan
    return finetune_for_evaluation(base, initial_classifier, memory, args)


def evaluate_paths(blocks, paths, args):
    loader = make_path_loader(paths, args.eval_batch, False, args.num_worker)
    return evaluate(blocks, loader, args)


def plasticity_summary(plasticity):
    completed = [row for row in plasticity.values() if len(row["ACC"]) == 3]
    if not completed:
        return {}
    return {
        "initial_acc": float(np.mean([row["ACC"][0] for row in completed])),
        "before_acc": float(np.mean([row["ACC"][1] for row in completed])),
        "after_acc": float(np.mean([row["ACC"][2] for row in completed])),
        "initial_mf1": float(np.mean([row["MF1"][0] for row in completed])),
        "before_mf1": float(np.mean([row["MF1"][1] for row in completed])),
        "after_mf1": float(np.mean([row["MF1"][2] for row in completed])),
    }


def write_metrics(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def run_progression(args, noise_rate: float, subjects: list[int]):
    fix_randomness(args.seed)
    variant_dir = args.output_root / "progression" / f"noise_{noise_rate:.2f}"
    initial_blocks, base, base_optimizer, memory = initialize_random_spr(args)
    inference_blocks = clone_blocks(initial_blocks, args)
    order = shuffled_subject_order(subjects, args.seed)
    if args.max_subjects > 0:
        order = order[: args.max_subjects]
    split_cache = {subject: subject_train_test(args, subject) for subject in order}
    anchors = order[: min(args.anchor_subjects, len(order))]
    performance = {
        "protocol": "task_progression",
        "random_initialization": True,
        "noise_rate": noise_rate,
        "order": order,
        "anchor_subjects": anchors,
        "completed_subjects": 0,
        "plasticity": {str(subject): {"ACC": [], "MF1": []} for subject in order},
        "anchor_history": {str(subject): [] for subject in anchors},
        "steps": [],
    }

    for step, subject in enumerate(order, start=1):
        train_paths, test_paths = split_cache[subject]
        previous_blocks = clone_blocks(inference_blocks, args)
        initial_result = evaluate_paths(initial_blocks, test_paths, args)
        before_result = evaluate_paths(previous_blocks, test_paths, args)
        train_metrics = train_stream_chunk(
            args, base, base_optimizer, memory, train_paths, noise_rate, step
        )
        inference_blocks, ft_loss = make_inference(args, base, initial_blocks[2], memory)
        after_result = evaluate_paths(inference_blocks, test_paths, args)
        performance["plasticity"][str(subject)] = {
            "ACC": [initial_result["acc"], before_result["acc"], after_result["acc"]],
            "MF1": [initial_result["mf1"], before_result["mf1"], after_result["mf1"]],
        }
        anchor_rows = []
        for anchor in anchors:
            if order.index(anchor) >= step:
                continue
            anchor_result = evaluate_paths(inference_blocks, split_cache[anchor][1], args)
            row = {"step": step, "acc": anchor_result["acc"], "mf1": anchor_result["mf1"]}
            performance["anchor_history"][str(anchor)].append(row)
            anchor_rows.append(row)
        step_row = {
            "step": step,
            "subject": subject,
            "adapt_sequences": len(train_paths[0]),
            "test_sequences": len(test_paths[0]),
            "current_initial_acc": initial_result["acc"],
            "current_before_acc": before_result["acc"],
            "current_after_acc": after_result["acc"],
            "current_after_mf1": after_result["mf1"],
            "anchor_mean_acc": float(np.mean([row["acc"] for row in anchor_rows])) if anchor_rows else math.nan,
            "anchor_mean_mf1": float(np.mean([row["mf1"] for row in anchor_rows])) if anchor_rows else math.nan,
            "finetune_loss": ft_loss,
            **train_metrics,
        }
        performance["steps"].append(step_row)
        performance["completed_subjects"] = step
        payload = {"performance": performance, "summary": plasticity_summary(performance["plasticity"])}
        write_metrics(variant_dir / "metrics.json", payload)
        print(
            f"[progression noise={noise_rate:.2f}] {step}/{len(order)} subject={subject} "
            f"current before/after={before_result['acc']:.4f}/{after_result['acc']:.4f} "
            f"anchor={step_row['anchor_mean_acc']:.4f} P={len(memory)}",
            flush=True,
        )
    return payload


def append_stability(performance, result):
    stability = performance["stability"]
    stability["ACC"].append(result["acc"])
    stability["MF1"].append(result["mf1"])
    stability["AAA"].append(float(compute_aaa(stability["ACC"])))
    stability["AAF1"] = compute_aaf1(stability["MF1"])
    stability["FR"].append(float(compute_forget(stability["ACC"])))


def run_fixed_split(args, noise_rate: float, subjects: list[int]):
    fix_randomness(args.seed)
    variant_dir = args.output_root / "fixed_split" / f"noise_{noise_rate:.2f}"
    train_idx, val_idx, old_idx, new_idx = split_subjects(subjects, args.seed)
    warmup_order = [int(subject) for subject in train_idx]
    if args.max_warmup_subjects > 0:
        warmup_order = warmup_order[: args.max_warmup_subjects]
    new_order = [int(subject) for subject in new_idx]
    if args.max_subjects > 0:
        new_order = new_order[: args.max_subjects]
    initial_blocks, base, base_optimizer, memory = initialize_random_spr(args)
    stream_step = 0
    warmup_rows = []
    for subject in warmup_order:
        stream_step += 1
        train_paths, _test_paths = subject_train_test(args, subject)
        metrics = train_stream_chunk(
            args, base, base_optimizer, memory, train_paths, noise_rate, stream_step
        )
        warmup_rows.append({"stream_step": stream_step, "subject": subject, **metrics})
        print(
            f"[fixed warmup noise={noise_rate:.2f}] {stream_step}/{len(warmup_order)} "
            f"subject={subject} P={len(memory)}",
            flush=True,
        )

    inference_blocks, warmup_ft_loss = make_inference(args, base, initial_blocks[2], memory)
    old_loader = make_loader(args.data_root, sorted(old_idx), args.eval_batch, False, args.num_worker)
    random_old = evaluate(initial_blocks, old_loader, args)
    warmup_old = evaluate(inference_blocks, old_loader, args)
    performance = {
        "protocol": "fixed_subject_split",
        "random_initialization": True,
        "noise_rate": noise_rate,
        "split": {
            "warmup_stream": warmup_order,
            "val_unused": sorted(int(subject) for subject in val_idx),
            "old_generalization": sorted(int(subject) for subject in old_idx),
            "new_order": [int(subject) for subject in new_order],
        },
        "random_initial_old": {"acc": random_old["acc"], "mf1": random_old["mf1"]},
        "warmup_finetune_loss": warmup_ft_loss,
        "warmup_steps": warmup_rows,
        "completed_subjects": 0,
        "stability": {"ACC": [], "MF1": [], "AAA": [], "AAF1": [], "FR": []},
        "plasticity": {str(subject): {"ACC": [], "MF1": []} for subject in new_order},
        "steps": [],
    }
    append_stability(performance, warmup_old)

    for index, subject in enumerate(new_order, start=1):
        stream_step += 1
        train_paths, test_paths = subject_train_test(args, subject)
        previous_blocks = clone_blocks(inference_blocks, args)
        initial_result = evaluate_paths(initial_blocks, test_paths, args)
        before_result = evaluate_paths(previous_blocks, test_paths, args)
        train_metrics = train_stream_chunk(
            args, base, base_optimizer, memory, train_paths, noise_rate, stream_step
        )
        inference_blocks, ft_loss = make_inference(args, base, initial_blocks[2], memory)
        after_result = evaluate_paths(inference_blocks, test_paths, args)
        old_result = evaluate(inference_blocks, old_loader, args)
        performance["plasticity"][str(subject)] = {
            "ACC": [initial_result["acc"], before_result["acc"], after_result["acc"]],
            "MF1": [initial_result["mf1"], before_result["mf1"], after_result["mf1"]],
        }
        append_stability(performance, old_result)
        step_row = {
            "step": index,
            "stream_step": stream_step,
            "subject": int(subject),
            "adapt_sequences": len(train_paths[0]),
            "test_sequences": len(test_paths[0]),
            "old_acc": old_result["acc"],
            "old_mf1": old_result["mf1"],
            "new_initial_acc": initial_result["acc"],
            "new_before_acc": before_result["acc"],
            "new_after_acc": after_result["acc"],
            "new_after_mf1": after_result["mf1"],
            "finetune_loss": ft_loss,
            **train_metrics,
        }
        performance["steps"].append(step_row)
        performance["completed_subjects"] = index
        payload = {"performance": performance, "summary": plasticity_summary(performance["plasticity"])}
        write_metrics(variant_dir / "metrics.json", payload)
        print(
            f"[fixed noise={noise_rate:.2f}] {index}/{len(new_order)} subject={subject} "
            f"old={old_result['acc']:.4f} new before/after={before_result['acc']:.4f}/{after_result['acc']:.4f} "
            f"P={len(memory)}",
            flush=True,
        )
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["progression", "fixed_split", "both"], default="both")
    parser.add_argument("--data-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"))
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "rttdp_brainuicl_runs" / "spr_random_init")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=10)
    parser.add_argument("--max-warmup-subjects", type=int, default=0)
    parser.add_argument("--anchor-subjects", type=int, default=3)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.0, 0.4])
    parser.add_argument("--expert-epochs", type=int, default=10)
    parser.add_argument("--base-epochs", type=int, default=10)
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--max-ssl-batches", type=int, default=0)
    parser.add_argument("--max-ft-batches", type=int, default=0)
    parser.add_argument("--ssl-lr", type=float, default=1e-5)
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--jitter", type=float, default=0.01)
    parser.add_argument("--mask-ratio", type=float, default=0.1)
    parser.add_argument("--channel-drop", type=float, default=0.1)
    parser.add_argument("--ensembles", type=int, default=5)
    parser.add_argument("--bmm-iters", type=int, default=10)
    parser.add_argument("--memory-capacity", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    fix_randomness(args.seed)
    args.dataset = "ISRUC"
    args.model_param = ModelConfig(args.dataset)
    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    subjects = discover_subjects(args.data_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "random_initialization": True,
        "pretrained_checkpoint": False,
        "initial_purified_buffer_epochs": 0,
        "label_source": "observed ISRUC labels, optionally symmetrically corrupted",
        "holdout_ratio": args.holdout_ratio,
        "memory_capacity_epochs": args.memory_capacity,
        "protocol": args.protocol,
        "seed": args.seed,
    }
    write_metrics(args.output_root / "protocol.json", config)
    results = {}
    for noise_rate in args.noise_rates:
        if args.protocol in ("progression", "both"):
            results[f"progression_noise_{noise_rate:.2f}"] = run_progression(args, noise_rate, subjects)["summary"]
        if args.protocol in ("fixed_split", "both"):
            results[f"fixed_split_noise_{noise_rate:.2f}"] = run_fixed_split(args, noise_rate, subjects)["summary"]
    write_metrics(args.output_root / "summary.json", {"config": config, "results": results})


if __name__ == "__main__":
    main()
