#!/usr/bin/env python3
"""Pretrain FACED with the same subject split and supervised BrainUICL flow."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from regularization_cl_eeg import build_split  # noqa: E402
from rttdp_brainuicl_full import (  # noqa: E402
    build_blocks,
    evaluate,
    flat_labels,
    flat_logits,
    forward_blocks,
    make_loader,
    save_blocks,
    set_train,
)
from utils.config import ModelConfig  # noqa: E402
from utils.util import fix_randomness  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    args = parser.parse_args()
    args.dataset = "FACED"
    args.max_subjects = 0
    args.model_param = ModelConfig(args.dataset)
    args.device = torch.device(
        f"cuda:{args.gpu}"
        if args.gpu >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    fix_randomness(args.seed)
    split = build_split(args)
    if not (
        len(split["train_idx"]) == 30
        and len(split["val_idx"]) == 8
        and len(split["old_idx"]) == 24
        and len(split["new_order"]) == 61
    ):
        raise RuntimeError(f"Unexpected FACED split sizes: {split}")

    train_loader = make_loader(
        args.data_root,
        split["train_idx"],
        args.batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    val_loader = make_loader(
        args.data_root,
        split["val_idx"],
        args.batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    blocks = build_blocks(args)
    optimizer = torch.optim.Adam(
        [parameter for block in blocks for parameter in block.parameters()],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    best_acc = -1.0
    best_epoch = 0
    best_blocks = None
    history = []
    for epoch in range(1, args.epochs + 1):
        set_train(blocks, True)
        losses = []
        for eog, eeg, labels in train_loader:
            logits = forward_blocks(
                blocks,
                eog.to(args.device),
                eeg.to(args.device),
                args,
            )
            loss = F.cross_entropy(flat_logits(logits), flat_labels(labels.to(args.device)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for block in blocks:
                torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(blocks, val_loader, args)
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "val_acc": validation["acc"],
            "val_mf1": validation["mf1"],
        }
        history.append(row)
        print(
            f"epoch={epoch}/{args.epochs} loss={row['train_loss']:.6f} "
            f"val_acc={row['val_acc']:.6f} val_mf1={row['val_mf1']:.6f}",
            flush=True,
        )
        if validation["acc"] > best_acc:
            best_acc = validation["acc"]
            best_epoch = epoch
            best_blocks = copy.deepcopy(blocks)

    checkpoint_dir = args.checkpoint_root / "FACED" / "Pretrain"
    save_blocks(best_blocks, checkpoint_dir, args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "split.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )
    (args.output_root / "pretrain_metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_acc": best_acc,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
