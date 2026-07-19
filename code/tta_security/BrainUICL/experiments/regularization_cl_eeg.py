#!/usr/bin/env python3
"""Clean, unlabeled, regularization-only continual learning on ISRUC EEG.

The guiding model supplies hard pseudo labels for every target epoch. The
student sees only the current subject: there is no confidence filtering,
replay buffer, DCB, or CEA update in this runner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.regularization_cl import (  # noqa: E402
    QuadraticImportanceStrategy,
    SynapticIntelligenceStrategy,
    build_regularization_strategy,
    freeze_batch_norm_running_stats,
    hard_pseudo_label_loss,
    named_trainable_parameters,
)
from regularization_cl_attacks import (  # noqa: E402
    brainwash_one_step_batch,
    materialize_poisoned_subject,
    pacol_gradient_matching_batch,
)
from rttdp_brainuicl_full import (  # noqa: E402
    CPCProbe,
    SequenceDataset,
    clone_blocks,
    discover_subjects,
    evaluate,
    flat_labels,
    flat_logits,
    forward_blocks,
    load_pretrained,
    make_loader,
    merge_subject_paths,
    save_blocks,
    set_train,
    split_subjects,
    subject_paths,
)
from utils.config import ModelConfig  # noqa: E402
from utils.util import compute_aaf1, compute_aaa, compute_forget, fix_randomness  # noqa: E402


METHODS = ("finetune", "ewc", "online_ewc", "si", "mas")
ATTACK_MODES = ("none", "pacol", "brainwash_reckless", "brainwash_cautious")


def parse_int_set(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def resolve_attack_tasks(value: str, total_tasks: int) -> set[int]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    resolved: set[int] = set()
    for item in items:
        task = total_tasks if item == "last" else int(item)
        if task < 1 or task > total_tasks:
            raise ValueError(
                f"Attack task {task} is outside the stream range 1..{total_tasks}"
            )
        resolved.add(task)
    return resolved


def make_subject_loader(args, subject: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SequenceDataset(subject_paths(args.data_root, subject)),
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=args.num_worker,
    )


def metric_view(result: dict) -> dict[str, float | int]:
    return {
        "acc": float(result["acc"]),
        "mf1": float(result["mf1"]),
        "n_epochs": int(result["n_epochs"]),
    }


@torch.no_grad()
def pseudo_label_diagnostics(guiding_blocks, loader, args) -> dict:
    set_train(guiding_blocks, False)
    y_true: list[int] = []
    y_pseudo: list[int] = []
    confidences: list[float] = []
    class_counts = np.zeros(args.model_param.NumClasses, dtype=np.int64)

    for eog, eeg, labels in loader:
        eog = eog.to(args.device)
        eeg = eeg.to(args.device)
        logits = forward_blocks(guiding_blocks, eog, eeg, args)
        probabilities = logits.softmax(dim=1)
        confidence, pseudo = probabilities.max(dim=1)
        pseudo_flat = pseudo.reshape(-1).detach().cpu().numpy()
        labels_flat = labels.reshape(-1).long().numpy()
        y_true.extend(labels_flat.tolist())
        y_pseudo.extend(pseudo_flat.tolist())
        confidences.extend(confidence.reshape(-1).detach().cpu().tolist())
        class_counts += np.bincount(
            pseudo_flat,
            minlength=args.model_param.NumClasses,
        )

    return {
        "coverage": 1.0,
        "acc_diagnostic_only": float(accuracy_score(y_true, y_pseudo)),
        "mf1_diagnostic_only": float(
            f1_score(
                y_true,
                y_pseudo,
                average="macro",
                labels=list(range(args.model_param.NumClasses)),
                zero_division=0,
            )
        ),
        "mean_confidence_diagnostic_only": float(np.mean(confidences)),
        "pseudo_class_counts": class_counts.astype(int).tolist(),
        "n_epochs": len(y_pseudo),
    }


def adapt_guiding_model(student_blocks, loader, args, task_index: int, subject: int):
    guiding_blocks = clone_blocks(student_blocks, args)
    if args.ssl_epoch <= 0:
        return guiding_blocks, []

    cpc = CPCProbe(guiding_blocks, args)
    epoch_losses: list[float] = []
    for epoch in range(1, args.ssl_epoch + 1):
        set_train(guiding_blocks, True)
        losses: list[float] = []
        for eog, eeg, _labels in loader:
            eog = eog.to(args.device)
            eeg = eeg.to(args.device)
            loss, guiding_blocks = cpc.update(eeg, eog)
            losses.append(float(loss))
        mean_loss = float(np.mean(losses))
        epoch_losses.append(mean_loss)
        print(
            f"[guide] task={task_index} subject={subject} "
            f"epoch={epoch}/{args.ssl_epoch} cpc={mean_loss:.6f}",
            flush=True,
        )
    return guiding_blocks, epoch_losses


def estimate_diagonal_importance(
    student_blocks,
    guiding_blocks,
    loader,
    args,
    mode: str,
) -> dict[str, torch.Tensor]:
    if mode not in {"fisher", "mas"}:
        raise ValueError(f"Unsupported importance mode: {mode}")

    parameters = named_trainable_parameters(student_blocks)
    tensors = [parameter for _name, parameter in parameters]
    importance = {
        name: torch.zeros_like(parameter)
        for name, parameter in parameters
    }
    set_train(student_blocks, False)
    set_train(guiding_blocks, False)
    batches = 0

    for eog, eeg, _labels in loader:
        eog = eog.to(args.device)
        eeg = eeg.to(args.device)
        student_logits = flat_logits(forward_blocks(student_blocks, eog, eeg, args))
        if mode == "fisher":
            with torch.no_grad():
                guiding_logits = flat_logits(
                    forward_blocks(guiding_blocks, eog, eeg, args)
                )
                pseudo_labels = guiding_logits.argmax(dim=1)
            objective = F.cross_entropy(student_logits, pseudo_labels)
        else:
            objective = 0.5 * student_logits.pow(2).sum(dim=1).mean()

        gradients = torch.autograd.grad(
            objective,
            tensors,
            allow_unused=True,
        )
        for (name, _parameter), gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            if mode == "fisher":
                importance[name].add_(gradient.detach().pow(2))
            else:
                importance[name].add_(gradient.detach().abs())
        batches += 1

    if batches == 0:
        raise RuntimeError("Cannot estimate importance from an empty subject loader")
    for name in importance:
        importance[name].div_(batches)
    for _name, parameter in parameters:
        parameter.grad = None
    return importance


def importance_summary(importance: dict[str, torch.Tensor] | None) -> dict:
    if not importance:
        return {"mean": 0.0, "max": 0.0, "nonzero_fraction": 0.0}
    total = 0
    nonzero = 0
    value_sum = 0.0
    value_max = 0.0
    for tensor in importance.values():
        detached = tensor.detach()
        total += detached.numel()
        nonzero += int((detached > 0).sum().item())
        value_sum += float(detached.sum().item())
        value_max = max(value_max, float(detached.max().item()))
    return {
        "mean": value_sum / max(total, 1),
        "max": value_max,
        "nonzero_fraction": nonzero / max(total, 1),
    }


def train_student_task(
    student_blocks,
    guiding_blocks,
    loader,
    strategy,
    args,
    task_index: int,
    subject: int,
) -> tuple[dict, dict]:
    parameters = named_trainable_parameters(student_blocks)
    optimizer = torch.optim.Adam(
        [parameter for _name, parameter in parameters],
        lr=args.cl_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    strategy.begin_task(parameters)
    epoch_rows: list[dict] = []

    for epoch in range(1, args.incremental_epoch + 1):
        set_train(student_blocks, True)
        if args.freeze_bn_stats:
            freeze_batch_norm_running_stats(student_blocks)
        set_train(guiding_blocks, False)
        pseudo_losses: list[float] = []
        regularization_losses: list[float] = []
        total_losses: list[float] = []

        for eog, eeg, _labels in loader:
            eog = eog.to(args.device)
            eeg = eeg.to(args.device)
            with torch.no_grad():
                guiding_logits = flat_logits(
                    forward_blocks(guiding_blocks, eog, eeg, args)
                )
            student_logits = flat_logits(
                forward_blocks(student_blocks, eog, eeg, args)
            )
            pseudo_loss, _pseudo_labels = hard_pseudo_label_loss(
                student_logits,
                guiding_logits,
            )
            regularization_loss = strategy.penalty(parameters)
            loss = pseudo_loss + regularization_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _name, parameter in parameters],
                    args.grad_clip,
                )
            strategy.capture_step(parameters)
            optimizer.step()
            strategy.finish_step(parameters)

            pseudo_losses.append(float(pseudo_loss.detach().cpu()))
            regularization_losses.append(
                float(regularization_loss.detach().cpu())
            )
            total_losses.append(float(loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "pseudo_loss": float(np.mean(pseudo_losses)),
            "regularization_loss": float(np.mean(regularization_losses)),
            "total_loss": float(np.mean(total_losses)),
        }
        epoch_rows.append(row)
        print(
            f"[student:{strategy.method}] task={task_index} subject={subject} "
            f"epoch={epoch}/{args.incremental_epoch} "
            f"pseudo={row['pseudo_loss']:.6f} "
            f"reg={row['regularization_loss']:.6f}",
            flush=True,
        )

    estimated_importance = None
    if isinstance(strategy, QuadraticImportanceStrategy):
        mode = "mas" if strategy.method == "mas" else "fisher"
        estimated_importance = estimate_diagonal_importance(
            student_blocks,
            guiding_blocks,
            loader,
            args,
            mode,
        )
    strategy.consolidate(parameters, estimated_importance)

    consolidated = getattr(strategy, "importance", None)
    return {
        "epochs": epoch_rows,
        "last_pseudo_loss": epoch_rows[-1]["pseudo_loss"],
        "last_regularization_loss": epoch_rows[-1]["regularization_loss"],
    }, importance_summary(consolidated)


def evaluate_seen_subjects(blocks, subjects, args) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for subject in subjects:
        loader = make_subject_loader(args, int(subject), shuffle=False)
        results[str(int(subject))] = metric_view(
            evaluate(blocks, loader, args, max_batches=args.eval_max_batches)
        )
    return results


def summarize_run(performance: dict) -> dict:
    task_rows = performance["tasks"]
    old_acc = performance["stability"]["acc"]
    old_mf1 = performance["stability"]["mf1"]
    before_acc = [row["current_before"]["acc"] for row in task_rows]
    after_acc = [row["current_after"]["acc"] for row in task_rows]
    before_mf1 = [row["current_before"]["mf1"] for row in task_rows]
    after_mf1 = [row["current_after"]["mf1"] for row in task_rows]
    final_seen = performance["final"]["seen_subjects"]

    final_seen_acc = [row["acc"] for row in final_seen.values()]
    final_seen_mf1 = [row["mf1"] for row in final_seen.values()]
    learned_acc = {
        str(row["subject"]): row["current_after"]["acc"] for row in task_rows
    }
    learned_mf1 = {
        str(row["subject"]): row["current_after"]["mf1"] for row in task_rows
    }
    bwt_acc = [
        final_seen[subject]["acc"] - learned_acc[subject]
        for subject in final_seen
    ]
    bwt_mf1 = [
        final_seen[subject]["mf1"] - learned_mf1[subject]
        for subject in final_seen
    ]

    return {
        "final_old_acc": old_acc[-1],
        "final_old_mf1": old_mf1[-1],
        "old_aaa": float(compute_aaa(old_acc)),
        "old_aaf1": float(compute_aaf1(old_mf1)[-1]),
        "old_fr": float(compute_forget(old_acc)),
        "mean_current_before_acc": float(np.mean(before_acc)),
        "mean_current_after_acc": float(np.mean(after_acc)),
        "mean_current_acc_gain": float(np.mean(np.subtract(after_acc, before_acc))),
        "mean_current_before_mf1": float(np.mean(before_mf1)),
        "mean_current_after_mf1": float(np.mean(after_mf1)),
        "mean_current_mf1_gain": float(np.mean(np.subtract(after_mf1, before_mf1))),
        "final_seen_acc": float(np.mean(final_seen_acc)),
        "final_seen_mf1": float(np.mean(final_seen_mf1)),
        "bwt_acc": float(np.mean(bwt_acc)),
        "bwt_mf1": float(np.mean(bwt_mf1)),
        "mean_pseudo_acc_diagnostic_only": float(
            np.mean([row["pseudo_labels"]["acc_diagnostic_only"] for row in task_rows])
        ),
        "mean_pseudo_mf1_diagnostic_only": float(
            np.mean([row["pseudo_labels"]["mf1_diagnostic_only"] for row in task_rows])
        ),
        "pseudo_label_coverage": 1.0,
    }


def write_report(path: Path, method: str, performance: dict, summary: dict) -> None:
    lines = [
        f"# Regularization-only EEG CL: {method}\n",
        "\n",
        "Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.\n",
        "\n",
        "## Final summary\n",
        "\n",
        "| metric | value |\n",
        "|---|---:|\n",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value:.6f} |\n")

    lines.extend(
        [
            "\n",
            "## Per-subject adaptation\n",
            "\n",
            "| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |\n",
            "|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in performance["tasks"]:
        lines.append(
            f"| {row['task']} | {row['subject']} | "
            f"{row['current_before']['acc']:.4f} | {row['current_after']['acc']:.4f} | "
            f"{row['current_before']['mf1']:.4f} | {row['current_after']['mf1']:.4f} | "
            f"{row['pseudo_labels']['acc_diagnostic_only']:.4f} | "
            f"{row['pseudo_labels']['mf1_diagnostic_only']:.4f} |\n"
        )
    path.write_text("".join(lines))


def save_progress(method_dir: Path, performance: dict) -> None:
    (method_dir / "metrics.json").write_text(
        json.dumps(performance, indent=2, ensure_ascii=False)
    )


def run_method(args, method: str, split: dict) -> tuple[dict, dict]:
    fix_randomness(args.seed)
    method_dir = args.output_root / method
    method_dir.mkdir(parents=True, exist_ok=True)

    student_blocks = load_pretrained(args)
    initial_blocks = load_pretrained(args)
    strategy = build_regularization_strategy(
        method,
        ewc_strength=args.ewc_strength,
        online_ewc_strength=args.online_ewc_strength,
        online_ewc_decay=args.online_ewc_decay,
        si_strength=args.si_strength,
        si_xi=args.si_xi,
        mas_strength=args.mas_strength,
        mas_decay=args.mas_decay,
    )

    old_loader = make_loader(
        args.data_root,
        split["old_idx"],
        args.batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    source_loader = make_loader(
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
    attack_reference_data_paths = merge_subject_paths(
        args.data_root,
        split["train_idx"],
    )[0]

    initial_old = metric_view(
        evaluate(student_blocks, old_loader, args, max_batches=args.eval_max_batches)
    )
    initial_source = metric_view(
        evaluate(student_blocks, source_loader, args, max_batches=args.eval_max_batches)
    )
    initial_val = metric_view(
        evaluate(student_blocks, val_loader, args, max_batches=args.eval_max_batches)
    )
    performance = {
        "method": method,
        "protocol": {
            "pseudo_labels": "all hard argmax labels from the guiding model",
            "confidence_filter": False,
            "replay": False,
            "dcb": False,
            "cea": False,
            "student_batch_norm_running_stats": (
                "frozen" if args.freeze_bn_stats else "updated"
            ),
            "attack": args.attack_mode,
            "attack_tasks": sorted(args.attack_tasks),
            "attacker_reference_inputs_used_by_learner": False,
            "true_target_labels_used_for_training": False,
        },
        "config": vars_for_json(args),
        "initial": {
            "old_generalization": initial_old,
            "source_train": initial_source,
            "validation": initial_val,
        },
        "stability": {
            "acc": [initial_old["acc"]],
            "mf1": [initial_old["mf1"]],
        },
        "tasks": [],
        "retention_snapshots": {},
        "final": {},
    }

    if 0 in args.checkpoint_milestones:
        save_blocks(student_blocks, method_dir / "checkpoints" / "Pretrain", args.seed)

    seen_subjects: list[int] = []
    total_tasks = len(split["new_order"])
    for task_index, subject in enumerate(split["new_order"], start=1):
        # Importance estimation adds method-specific data passes. Resetting at
        # each task prevents those passes from changing the next task's loader,
        # dropout, and CPC sampling sequence across otherwise identical runs.
        fix_randomness(args.seed + 1000 * task_index)
        print(
            f"[{method}] task={task_index}/{total_tasks} subject={subject}",
            flush=True,
        )
        clean_paths = subject_paths(args.data_root, subject)
        loader_eval = make_subject_loader(args, subject, shuffle=False)
        before = metric_view(
            evaluate(student_blocks, loader_eval, args, max_batches=args.eval_max_batches)
        )

        attack_diagnostics = None
        attack_label_cpc_losses: list[float] = []
        if task_index in args.attack_tasks:
            clean_label_loader = make_subject_loader(args, subject, shuffle=True)
            label_blocks, attack_label_cpc_losses = adapt_guiding_model(
                student_blocks,
                clean_label_loader,
                args,
                task_index,
                subject,
            )
            if args.attack_mode == "pacol":
                attack_function = pacol_gradient_matching_batch
            elif args.attack_mode in {"brainwash_reckless", "brainwash_cautious"}:
                attack_function = brainwash_one_step_batch
            else:
                raise ValueError(
                    f"Attack tasks were supplied for unsupported mode {args.attack_mode}"
                )
            poisoned_data_paths, attack_diagnostics = materialize_poisoned_subject(
                attack=attack_function,
                student_blocks=student_blocks,
                label_blocks=label_blocks,
                current_data_paths=clean_paths[0],
                reference_data_paths=attack_reference_data_paths,
                output_dir=(
                    method_dir
                    / "poisoned_inputs"
                    / f"task_{task_index}_subject_{subject}"
                ),
                task_index=task_index,
                subject=subject,
                args=args,
            )
            training_paths = (poisoned_data_paths, clean_paths[1])
            # Attack generation has extra model/data passes. Restore the task
            # seed so the formal guide/student run remains method-comparable.
            fix_randomness(args.seed + 1000 * task_index)
            loader_train = DataLoader(
                SequenceDataset(training_paths),
                batch_size=args.batch,
                shuffle=True,
                num_workers=args.num_worker,
            )
            loader_pseudo_eval = DataLoader(
                SequenceDataset(training_paths),
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_worker,
            )
        else:
            loader_train = make_subject_loader(args, subject, shuffle=True)
            loader_pseudo_eval = loader_eval

        guiding_blocks, cpc_losses = adapt_guiding_model(
            student_blocks,
            loader_train,
            args,
            task_index,
            subject,
        )
        pseudo_diagnostics = pseudo_label_diagnostics(
            guiding_blocks,
            loader_pseudo_eval,
            args,
        )
        clean_pseudo_diagnostics = (
            pseudo_label_diagnostics(guiding_blocks, loader_eval, args)
            if attack_diagnostics is not None
            else pseudo_diagnostics
        )
        train_diagnostics, current_importance = train_student_task(
            student_blocks,
            guiding_blocks,
            loader_train,
            strategy,
            args,
            task_index,
            subject,
        )

        after = metric_view(
            evaluate(student_blocks, loader_eval, args, max_batches=args.eval_max_batches)
        )
        old = metric_view(
            evaluate(student_blocks, old_loader, args, max_batches=args.eval_max_batches)
        )
        performance["stability"]["acc"].append(old["acc"])
        performance["stability"]["mf1"].append(old["mf1"])
        seen_subjects.append(int(subject))

        task_row = {
            "task": task_index,
            "subject": int(subject),
            "current_before": before,
            "current_after": after,
            "old_generalization_after": old,
            "pseudo_labels": pseudo_diagnostics,
            "pseudo_labels_on_clean_current": clean_pseudo_diagnostics,
            "guiding_cpc_losses": cpc_losses,
            "attack_label_guiding_cpc_losses": attack_label_cpc_losses,
            "attack": attack_diagnostics,
            "training": train_diagnostics,
            "importance": current_importance,
        }
        performance["tasks"].append(task_row)

        if task_index in args.retention_milestones or task_index == total_tasks:
            performance["retention_snapshots"][str(task_index)] = (
                evaluate_seen_subjects(student_blocks, seen_subjects, args)
            )
        if task_index in args.checkpoint_milestones or task_index == total_tasks:
            checkpoint_dir = method_dir / "checkpoints" / f"individual_{task_index}"
            save_blocks(student_blocks, checkpoint_dir, args.seed)
            torch.save(strategy.state_dict(), checkpoint_dir / "regularizer_state.pt")

        save_progress(method_dir, performance)
        print(
            f"[{method}] subject={subject} current "
            f"ACC {before['acc']:.4f}->{after['acc']:.4f} "
            f"MF1 {before['mf1']:.4f}->{after['mf1']:.4f}; "
            f"old ACC={old['acc']:.4f} MF1={old['mf1']:.4f}; "
            f"pseudo ACC={pseudo_diagnostics['acc_diagnostic_only']:.4f}",
            flush=True,
        )

    final_seen = performance["retention_snapshots"][str(total_tasks)]
    performance["final"] = {
        "old_generalization": metric_view(
            evaluate(student_blocks, old_loader, args, max_batches=args.eval_max_batches)
        ),
        "source_train": metric_view(
            evaluate(student_blocks, source_loader, args, max_batches=args.eval_max_batches)
        ),
        "validation": metric_view(
            evaluate(student_blocks, val_loader, args, max_batches=args.eval_max_batches)
        ),
        "seen_subjects": final_seen,
        "initial_model_seen_subjects": evaluate_seen_subjects(
            initial_blocks,
            split["new_order"],
            args,
        ),
    }
    summary = summarize_run(performance)
    performance["summary"] = summary
    save_progress(method_dir, performance)
    write_report(method_dir / "report.md", method, performance, summary)
    return performance, summary


def vars_for_json(args) -> dict:
    output = {}
    for key, value in vars(args).items():
        if key in {"device", "model_param"}:
            output[key] = str(value)
        elif isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, set):
            output[key] = sorted(value)
        else:
            output[key] = value
    return output


def build_split(args) -> dict:
    subjects = discover_subjects(args.data_root)
    train_idx, val_idx, old_idx, new_idx = split_subjects(subjects, args.seed)
    new_order = [int(subject) for subject in new_idx]
    if args.max_subjects > 0:
        new_order = new_order[: args.max_subjects]
    return {
        "seed": args.seed,
        "train_idx": [int(subject) for subject in train_idx],
        "val_idx": [int(subject) for subject in val_idx],
        "old_idx": [int(subject) for subject in old_idx],
        "new_order": new_order,
        "full_new_order": [int(subject) for subject in new_idx],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/home/undefined/Disk/datasets/brainuicl/processed/"
            "isruc_group1_npy_float32"
        ),
    )
    parser.add_argument(
        "--input-checkpoint-root",
        type=Path,
        default=Path("/home/undefined/Disk/ai-storage/BrainUICL/model_parameter"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "regularization_cl_eeg_runs" / "latest",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(METHODS),
        help="Comma-separated subset of finetune,ewc,online_ewc,si,mas.",
    )
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--ssl-epoch", type=int, default=10)
    parser.add_argument("--incremental-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssl-lr", type=float, default=1e-6)
    parser.add_argument("--cl-lr", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--freeze-bn-stats",
        action="store_true",
        help=(
            "Keep the pretrained student BatchNorm running mean/variance fixed "
            "during CL; BN affine parameters remain trainable."
        ),
    )
    parser.add_argument("--ewc-strength", type=float, default=5000.0)
    parser.add_argument("--online-ewc-strength", type=float, default=6500.0)
    parser.add_argument("--online-ewc-decay", type=float, default=1.0)
    parser.add_argument("--si-strength", type=float, default=15.0)
    parser.add_argument("--si-xi", type=float, default=0.1)
    parser.add_argument("--mas-strength", type=float, default=1.0)
    parser.add_argument("--mas-decay", type=float, default=1.0)
    parser.add_argument("--attack-mode", choices=ATTACK_MODES, default="none")
    parser.add_argument(
        "--attack-tasks",
        type=str,
        default="",
        help="Comma-separated 1-based task indices; 'last' selects the final task.",
    )
    parser.add_argument("--attack-fraction", type=float, default=0.05)
    parser.add_argument(
        "--attack-eps-scale",
        type=float,
        default=0.10,
        help="Per-modality L-infinity budget as a fraction of its signal std.",
    )
    parser.add_argument("--attack-steps", type=int, default=5)
    parser.add_argument("--attack-inner-lr", type=float, default=1e-4)
    parser.add_argument("--attack-cautious-weight", type=float, default=1.0)
    parser.add_argument(
        "--attack-param-scope",
        choices=("classifier", "encoder_head", "all"),
        default="classifier",
    )
    parser.add_argument("--attack-reference-batch", type=int, default=1)
    parser.add_argument("--attack-random-start", action="store_true")
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--retention-milestones", type=str, default="10,25,49")
    parser.add_argument("--checkpoint-milestones", type=str, default="0,1,10,25,49")
    return parser.parse_args()


def main():
    args = parse_args()
    requested_methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(requested_methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if not requested_methods:
        raise ValueError("At least one method is required")

    args.dataset = "ISRUC"
    args.model_param = ModelConfig(args.dataset)
    args.device = torch.device(
        f"cuda:{args.gpu}"
        if args.gpu >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    args.retention_milestones = parse_int_set(args.retention_milestones)
    args.checkpoint_milestones = parse_int_set(args.checkpoint_milestones)
    args.output_root.mkdir(parents=True, exist_ok=True)

    split = build_split(args)
    args.attack_tasks = resolve_attack_tasks(
        args.attack_tasks,
        len(split["new_order"]),
    )
    if args.attack_mode == "none" and args.attack_tasks:
        raise ValueError("--attack-tasks requires a non-'none' --attack-mode")
    if args.attack_mode != "none" and not args.attack_tasks:
        raise ValueError("A non-'none' --attack-mode requires --attack-tasks")
    if not 0.0 <= args.attack_fraction <= 1.0:
        raise ValueError("--attack-fraction must be in [0, 1]")
    if args.attack_steps < 1:
        raise ValueError("--attack-steps must be positive")
    if args.attack_reference_batch < 1:
        raise ValueError("--attack-reference-batch must be positive")
    (args.output_root / "split.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False)
    )
    (args.output_root / "config.json").write_text(
        json.dumps(vars_for_json(args), indent=2, ensure_ascii=False)
    )
    print(json.dumps(split, ensure_ascii=False), flush=True)

    summaries = {}
    for method in requested_methods:
        _performance, summary = run_method(args, method, split)
        summaries[method] = summary
        (args.output_root / "summary.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False)
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
