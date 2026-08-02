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
from model.icml2026_cl_defenses import (  # noqa: E402
    RobustFeatureDefense,
    T2TDetector,
    diagonal_t2t_score,
)
from n2n_shared_proxy import resolve_task as resolve_n2n_task  # noqa: E402
from unlabeled_eeg import UnlabeledSequenceDataset  # noqa: E402
from regularization_cl_attacks import (  # noqa: E402
    brainwash_one_step_batch,
    materialize_batched_proxy_dual_harm_subject,
    materialize_poisoned_subject,
    pacol_gradient_matching_batch,
)
from progressive_feedback_proxy import (  # noqa: E402
    ProgressiveFeedbackProxy,
    add_progressive_proxy_args,
    public_probabilities,
    validate_progressive_proxy_args,
)
from rttdp_brainuicl_full import (  # noqa: E402
    CPCProbe,
    SequenceDataset,
    clone_blocks,
    discover_subjects,
    evaluate,
    external_proxy_upload_paths,
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
ATTACK_MODES = (
    "none",
    "benign_repeat",
    "pacol",
    "brainwash_reckless",
    "brainwash_cautious",
    "proxy_dual_harm",
)
DEFENSE_MODES = ("none", "t2t", "robust_feature")
TASK_PHASE_OFFSETS = {"setup": 0, "guide": 1, "student": 2}


def task_phase_seed(seed: int, task_index: int, phase: str) -> int:
    try:
        offset = TASK_PHASE_OFFSETS[phase]
    except KeyError as error:
        raise ValueError(f"Unknown task phase: {phase}") from error
    return int(seed) + 1000 * int(task_index) + offset


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


def make_unlabeled_loader(args, data_paths: list[Path], shuffle: bool) -> DataLoader:
    return DataLoader(
        UnlabeledSequenceDataset(data_paths, args.model_param.SeqLength),
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=args.num_worker,
    )


def resolve_n2n_subject_paths(
    args,
    task_index: int,
    subject: int,
    clean_paths: tuple[list[Path], list[Path]],
) -> tuple[tuple[list[Path], list[Path]], dict]:
    """Replace only manifest-selected signals and preserve clean annotations."""

    resolved = resolve_n2n_task(
        args.n2n_manifest,
        task=task_index,
        subject=subject,
        clean_data_paths=clean_paths[0],
        verify=args.n2n_verify,
    )
    training_paths = (list(resolved.data_paths), list(clean_paths[1]))
    return training_paths, dict(resolved.diagnostics)


def metric_view(result: dict) -> dict[str, float | int]:
    return {
        "acc": float(result["acc"]),
        "mf1": float(result["mf1"]),
        "n_epochs": int(result["n_epochs"]),
    }


def parameter_names_for_scope(
    parameters: list[tuple[str, torch.nn.Parameter]],
    scope: str,
) -> set[str]:
    if scope == "all":
        return {name for name, _parameter in parameters}
    if scope == "head":
        return {
            name
            for name, _parameter in parameters
            if name.startswith("sleep_classifier.")
        }
    if scope == "classifier":
        return {
            name
            for name, _parameter in parameters
            if name.startswith("sleep_classifier.sleep_stage_classifier.")
        }
    raise ValueError(f"Unknown parameter scope: {scope}")


def snapshot_parameters(
    parameters: list[tuple[str, torch.nn.Parameter]],
    names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in parameters
        if names is None or name in names
    }


def snapshot_tensor_map(
    values: dict[str, torch.Tensor],
    names: set[str],
) -> dict[str, torch.Tensor]:
    return {
        name: values[name].detach().cpu().clone()
        for name in names
    }


def snapshot_blocks(blocks) -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(
        {
            name: tensor.detach().cpu().clone()
            for name, tensor in block.state_dict().items()
        }
        for block in blocks
    )


def restore_blocks(blocks, states: tuple[dict[str, torch.Tensor], ...]) -> None:
    if len(blocks) != len(states):
        raise ValueError("Block snapshot has an incompatible length")
    for block, state in zip(blocks, states):
        block.load_state_dict(state)


@torch.no_grad()
def estimate_classifier_feature_covariance(
    student_blocks,
    loader,
    args,
) -> tuple[torch.Tensor, int]:
    """Estimate X^T X/n for the final linear EEG classifier without labels."""
    set_train(student_blocks, False)
    classifier = student_blocks[2].sleep_stage_classifier
    feature_dim = classifier.in_features
    second_moment = torch.zeros(
        feature_dim,
        feature_dim,
        device=args.device,
        dtype=torch.float64,
    )
    sample_count = 0
    for eog, eeg, _labels in loader:
        eog = eog.to(args.device).reshape(
            -1,
            args.model_param.EogNum,
            args.model_param.EpochLength,
        )
        eeg = eeg.to(args.device).reshape(
            -1,
            args.model_param.EegNum,
            args.model_param.EpochLength,
        )
        features = student_blocks[0](eeg, eog)
        features = student_blocks[1](features)
        features = student_blocks[2].sleep_stage_mlp(features)
        features = features.reshape(-1, feature_dim).double()
        second_moment.add_(features.T @ features)
        sample_count += features.shape[0]
    if sample_count == 0:
        raise RuntimeError("Cannot estimate feature covariance from an empty task")
    covariance = (second_moment / sample_count).to(dtype=classifier.weight.dtype)
    return covariance, sample_count


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
    *,
    robust_feature_defense: RobustFeatureDefense | None = None,
    need_fisher_curvature: bool = False,
    curvature_loader=None,
) -> tuple[dict, dict, dict[str, torch.Tensor] | None]:
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
        defense_losses: list[float] = []
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
            defense_loss = (
                robust_feature_defense.penalty(
                    student_blocks[2].sleep_stage_classifier.weight
                )
                if robust_feature_defense is not None
                else pseudo_loss.new_zeros(())
            )
            loss = pseudo_loss + regularization_loss + defense_loss

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
            defense_losses.append(float(defense_loss.detach().cpu()))
            total_losses.append(float(loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "pseudo_loss": float(np.mean(pseudo_losses)),
            "regularization_loss": float(np.mean(regularization_losses)),
            "defense_loss": float(np.mean(defense_losses)),
            "total_loss": float(np.mean(total_losses)),
        }
        epoch_rows.append(row)
        print(
            f"[student:{strategy.method}] task={task_index} subject={subject} "
            f"epoch={epoch}/{args.incremental_epoch} "
            f"pseudo={row['pseudo_loss']:.6f} "
            f"reg={row['regularization_loss']:.6f} "
            f"defense={row['defense_loss']:.6f}",
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
    fisher_curvature = None
    if need_fisher_curvature:
        if (
            isinstance(strategy, QuadraticImportanceStrategy)
            and strategy.method in {"ewc", "online_ewc"}
        ):
            fisher_curvature = estimated_importance
        else:
            fisher_curvature = estimate_diagonal_importance(
                student_blocks,
                guiding_blocks,
                curvature_loader if curvature_loader is not None else loader,
                args,
                "fisher",
            )
    strategy.consolidate(parameters, estimated_importance)

    consolidated = getattr(strategy, "importance", None)
    return {
        "epochs": epoch_rows,
        "last_pseudo_loss": epoch_rows[-1]["pseudo_loss"],
        "last_regularization_loss": epoch_rows[-1]["regularization_loss"],
        "last_defense_loss": epoch_rows[-1]["defense_loss"],
    }, importance_summary(consolidated), fisher_curvature


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

    summary = {
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
    defense_mode = performance["protocol"].get("icml2026_defense", "none")
    if defense_mode == "t2t":
        decisions = [
            row["defense"]
            for row in task_rows
            if row.get("defense", {}).get("valid")
        ]
        summary.update(
            {
                "t2t_valid_scores": len(decisions),
                "t2t_detected_pairs": sum(
                    int(row.get("detected", False)) for row in decisions
                ),
                "t2t_rejected_updates": sum(
                    len(row.get("rejected_task_indices", [])) for row in decisions
                ),
            }
        )
    elif defense_mode == "robust_feature":
        rows = [row["defense"] for row in task_rows if row.get("defense")]
        summary.update(
            {
                "robust_mean_protected_fraction": float(
                    np.mean([row["protected_fraction"] for row in rows])
                ),
                "robust_mean_lambda": float(
                    np.mean([row["lambda_mean"] for row in rows])
                ),
                "robust_mean_defense_loss": float(
                    np.mean(
                        [row["training_last_defense_loss"] for row in rows]
                    )
                ),
            }
        )
    return summary


def write_report(path: Path, method: str, performance: dict, summary: dict) -> None:
    lines = [
        f"# Regularization-only EEG CL: {method}\n",
        "\n",
        "Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.\n",
        f"ICML 2026 defense mode: `{performance['protocol'].get('icml2026_defense', 'none')}`.\n",
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


def delete_generated_inputs(paths: list[Path], generated_root: Path) -> int:
    root = generated_root.resolve()
    deleted = 0
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Refusing to delete input outside {root}: {resolved}")
        if resolved.is_file():
            resolved.unlink()
            deleted += 1
    return deleted


def run_method(args, method: str, split: dict) -> tuple[dict, dict]:
    fix_randomness(args.seed)
    method_dir = args.output_root / method
    method_dir.mkdir(parents=True, exist_ok=True)

    student_blocks = load_pretrained(args)
    initial_blocks = load_pretrained(args)
    progressive_proxy = (
        ProgressiveFeedbackProxy(
            args,
            split,
            method_dir,
            retain_payloads_for_replay=False,
        )
        if args.progressive_proxy_mode != "none"
        else None
    )
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
    robust_feature_defense = (
        RobustFeatureDefense(
            sigma2=args.robust_feature_sigma2,
            budget_per_dimension=args.robust_feature_budget_per_dimension,
            initial_risk=args.robust_feature_initial_risk,
            eps=args.robust_feature_eps,
            max_regularizer=args.robust_feature_max_regularizer,
        )
        if args.defense_mode == "robust_feature"
        else None
    )
    t2t_detector = (
        T2TDetector(
            threshold_multiplier=args.t2t_threshold_multiplier,
            window=args.t2t_window,
            minimum_history=args.t2t_minimum_history,
            score_floor=args.t2t_score_floor,
        )
        if args.defense_mode == "t2t"
        else None
    )
    initial_parameters = named_trainable_parameters(student_blocks)
    t2t_names = (
        parameter_names_for_scope(initial_parameters, args.t2t_param_scope)
        if t2t_detector is not None
        else set()
    )
    t2t_dimension_count = sum(
        parameter.numel()
        for name, parameter in initial_parameters
        if name in t2t_names
    )
    t2t_history: list[dict] = []
    if t2t_detector is not None:
        t2t_history.append(
            {
                "task": 0,
                "parameters": snapshot_parameters(initial_parameters, t2t_names),
                "blocks": snapshot_blocks(student_blocks),
                "strategy": strategy.state_dict(),
                "hessian": None,
                "regularizer": None,
            }
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
            "external_proxy_noise": (
                str(args.noise_upload_root) if args.noise_upload_root is not None else None
            ),
            "canonical_n2n_manifest": (
                None if args.n2n_manifest is None else str(args.n2n_manifest)
            ),
            "upload_cardinality": "N-to-N" if args.n2n_manifest is not None else None,
            "target_training_loader": "signal-only without annotation paths",
            "proxy_reference_inputs_used_by_learner": False,
            "true_target_labels_used_for_training": False,
            "icml2026_defense": args.defense_mode,
            "t2t_parameter_scope": (
                args.t2t_param_scope if t2t_detector is not None else None
            ),
            "t2t_action": (
                args.t2t_action if t2t_detector is not None else None
            ),
            "robust_feature_scope": (
                "sleep_stage_classifier.weight"
                if robust_feature_defense is not None
                else None
            ),
            "progressive_feedback_proxy": (
                progressive_proxy.protocol()
                if progressive_proxy is not None
                else None
            ),
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
    if args.noise_upload_root is None and args.n2n_manifest is None:
        performance["protocol"].update(
            {
                "attack": args.attack_mode,
                "attack_tasks": sorted(args.attack_tasks),
                "attacker_reference_inputs_used_by_learner": False,
            }
        )

    if not args.no_save_checkpoints and 0 in args.checkpoint_milestones:
        save_blocks(student_blocks, method_dir / "checkpoints" / "Pretrain", args.seed)

    seen_subjects: list[int] = []
    total_tasks = len(split["new_order"])
    for task_index, subject in enumerate(split["new_order"], start=1):
        # Importance estimation adds method-specific data passes. Resetting at
        # each task prevents those passes from changing the next task's loader,
        # dropout, and CPC sampling sequence across otherwise identical runs.
        fix_randomness(task_phase_seed(args.seed, task_index, "setup"))
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
        noise_diagnostics = None
        attack_label_cpc_losses: list[float] = []
        generated_poisoned_paths: list[Path] = []
        if progressive_proxy is not None:
            progressive_data_paths, _tracked_paths, progressive_diagnostics = (
                progressive_proxy.prepare_task(
                    task_index,
                    subject,
                    clean_paths[0],
                )
            )
            training_paths = (
                progressive_data_paths,
                progressive_proxy.diagnostic_label_paths(
                    task_index,
                    clean_paths[1],
                ),
            )
            loader_train = make_unlabeled_loader(args, progressive_data_paths, True)
            loader_pseudo_eval = make_unlabeled_loader(
                args, progressive_data_paths, False
            )
            loader_diagnostic = DataLoader(
                SequenceDataset(training_paths),
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_worker,
            )
        elif args.n2n_manifest is not None:
            training_paths, noise_diagnostics = resolve_n2n_subject_paths(
                args,
                task_index,
                subject,
                clean_paths,
            )
            loader_train = make_unlabeled_loader(args, training_paths[0], True)
            loader_pseudo_eval = make_unlabeled_loader(args, training_paths[0], False)
            loader_diagnostic = DataLoader(
                SequenceDataset(training_paths),
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_worker,
            )
        elif args.noise_upload_root is not None:
            training_paths, noise_diagnostics = external_proxy_upload_paths(
                args.noise_upload_root,
                clean_paths,
                task_index,
                subject,
            )
            noise_diagnostics = dict(noise_diagnostics)
            noise_diagnostics["source"] = str(args.noise_upload_root)
            loader_train = make_unlabeled_loader(args, training_paths[0], True)
            loader_pseudo_eval = make_unlabeled_loader(args, training_paths[0], False)
            loader_diagnostic = DataLoader(
                SequenceDataset(training_paths),
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_worker,
            )
        elif task_index in args.attack_tasks:
            if args.attack_mode == "benign_repeat":
                poisoned_data_paths = list(clean_paths[0])
                attack_diagnostics = {
                    "mode": "benign_repeat",
                    "task": int(task_index),
                    "subject": int(subject),
                    "poisoned_sequences": 0,
                    "total_sequences": len(clean_paths[0]),
                    "poison_fraction": 0.0,
                    "poison_indices": list(range(len(clean_paths[0]))),
                    "generated_inputs": False,
                    "learner_replay": False,
                    "diagnostics_mean": {
                        "relative_l2_eog": 0.0,
                        "relative_l2_eeg": 0.0,
                        "pseudo_label_preservation": 1.0,
                    },
                }
            else:
                clean_label_loader = make_unlabeled_loader(
                    args, list(clean_paths[0]), shuffle=True
                )
                label_blocks, attack_label_cpc_losses = adapt_guiding_model(
                    student_blocks,
                    clean_label_loader,
                    args,
                    task_index,
                    subject,
                )
            if args.attack_mode == "proxy_dual_harm":
                poisoned_data_paths, attack_diagnostics = (
                    materialize_batched_proxy_dual_harm_subject(
                        student_blocks=student_blocks,
                        label_blocks=label_blocks,
                        strategy=strategy,
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
                )
            elif args.attack_mode != "benign_repeat":
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
            training_data_paths = list(poisoned_data_paths)
            training_label_paths = list(clean_paths[1])
            if attack_diagnostics.get("generated_inputs", True):
                generated_poisoned_paths = [
                    poisoned_data_paths[index]
                    for index in attack_diagnostics["poison_indices"]
                ]
            injected_proxy_sequences = 0
            if args.attack_proxy_repeat > 0 and attack_diagnostics is not None:
                repeat_indices = attack_diagnostics["poison_indices"]
                proxy_paths = [poisoned_data_paths[index] for index in repeat_indices]
                proxy_labels = [clean_paths[1][index] for index in repeat_indices]
                training_data_paths.extend(proxy_paths * args.attack_proxy_repeat)
                training_label_paths.extend(proxy_labels * args.attack_proxy_repeat)
                injected_proxy_sequences = len(proxy_paths) * args.attack_proxy_repeat
                attack_diagnostics["injected_proxy_sequences"] = injected_proxy_sequences
                attack_diagnostics["training_sequences_after_injection"] = len(
                    training_data_paths
                )
                attack_diagnostics["proxy_repeat"] = args.attack_proxy_repeat
            training_paths = (training_data_paths, training_label_paths)
            loader_train = make_unlabeled_loader(args, training_data_paths, True)
            loader_pseudo_eval = make_unlabeled_loader(args, training_data_paths, False)
            loader_diagnostic = DataLoader(
                SequenceDataset(training_paths),
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_worker,
            )
        else:
            loader_train = make_unlabeled_loader(args, list(clean_paths[0]), True)
            loader_pseudo_eval = make_unlabeled_loader(
                args, list(clean_paths[0]), False
            )
            loader_diagnostic = loader_eval

        # Keep condition-specific manifest checks or generation passes from
        # changing the formal guide's loader order and stochastic layers.
        fix_randomness(task_phase_seed(args.seed, task_index, "guide"))
        guiding_blocks, cpc_losses = adapt_guiding_model(
            student_blocks,
            loader_train,
            args,
            task_index,
            subject,
        )
        pseudo_diagnostics = pseudo_label_diagnostics(
            guiding_blocks,
            loader_diagnostic,
            args,
        )
        clean_pseudo_diagnostics = (
            pseudo_label_diagnostics(guiding_blocks, loader_eval, args)
            if attack_diagnostics is not None or noise_diagnostics is not None
            else pseudo_diagnostics
        )
        task_parameters = named_trainable_parameters(student_blocks)
        t2t_regularizer = None
        if t2t_detector is not None:
            t2t_regularizer = snapshot_tensor_map(
                strategy.curvature(task_parameters),
                t2t_names,
            )

        defense_diagnostics = None
        if robust_feature_defense is not None:
            feature_covariance, feature_sample_count = (
                estimate_classifier_feature_covariance(
                    student_blocks,
                    loader_pseudo_eval,
                    args,
                )
            )
            defense_diagnostics = robust_feature_defense.prepare_task(
                student_blocks[2].sleep_stage_classifier.weight,
                feature_covariance,
                feature_sample_count,
            )

        # Even a non-shuffled DataLoader iterator consumes torch RNG state.
        # Isolate optimization from the extra clean-input diagnostic in proxy runs.
        fix_randomness(task_phase_seed(args.seed, task_index, "student"))
        train_diagnostics, current_importance, task_fisher = train_student_task(
            student_blocks,
            guiding_blocks,
            loader_train,
            strategy,
            args,
            task_index,
            subject,
            robust_feature_defense=robust_feature_defense,
            need_fisher_curvature=t2t_detector is not None,
            curvature_loader=loader_pseudo_eval,
        )

        progressive_task_row = None
        if progressive_proxy is not None and (
            progressive_proxy.is_proxy_task(task_index)
            or progressive_proxy.is_clean_feedback_task(task_index)
        ):
            response_probabilities = public_probabilities(
                student_blocks,
                training_paths[0],
                args,
            )
            progressive_task_row = progressive_proxy.observe_task(
                task_index,
                subject,
                training_paths[0],
                response_probabilities,
            )

        if robust_feature_defense is not None:
            defense_diagnostics = robust_feature_defense.finish_task()
            defense_diagnostics["training_last_defense_loss"] = train_diagnostics[
                "last_defense_loss"
            ]

        if t2t_detector is not None:
            if task_fisher is None or t2t_regularizer is None:
                raise RuntimeError("T2T requires task Fisher and regularizer curvature")
            current_parameters = named_trainable_parameters(student_blocks)
            current_parameter_snapshot = snapshot_parameters(
                current_parameters,
                t2t_names,
            )
            current_hessian = snapshot_tensor_map(task_fisher, t2t_names)
            if len(t2t_history) >= 2:
                previous = t2t_history[-1]
                previous_previous = t2t_history[-2]
                score = diagonal_t2t_score(
                    current_parameter_snapshot,
                    previous["parameters"],
                    previous_previous["parameters"],
                    current_hessian,
                    previous["hessian"],
                    t2t_regularizer,
                    previous["regularizer"],
                    pinv_rtol=args.t2t_pinv_rtol,
                    eps=args.t2t_eps,
                )
                defense_diagnostics = t2t_detector.decide(score)
                if not score["valid"]:
                    defense_diagnostics["reason"] = "empty common diagonal subspace"
            else:
                defense_diagnostics = {
                    "valid": False,
                    "score": 0.0,
                    "score_rms": 0.0,
                    "active_dimensions": 0,
                    "total_dimensions": t2t_dimension_count,
                    "active_fraction": 0.0,
                    "moving_mean": None,
                    "threshold": None,
                    "detected": False,
                    "reason": "two previous model states are not yet available",
                }

            should_rollback = bool(
                defense_diagnostics["detected"]
                and args.t2t_action == "rollback"
            )
            defense_diagnostics["action"] = (
                "rollback"
                if should_rollback
                else "monitor_only"
                if defense_diagnostics["detected"]
                else "accept"
            )
            if should_rollback:
                provisional_current = metric_view(
                    evaluate(
                        student_blocks,
                        loader_eval,
                        args,
                        max_batches=args.eval_max_batches,
                    )
                )
                provisional_old = metric_view(
                    evaluate(
                        student_blocks,
                        old_loader,
                        args,
                        max_batches=args.eval_max_batches,
                    )
                )
                rollback_state = t2t_history[-2]
                rejected_tasks = [int(t2t_history[-1]["task"]), task_index]
                restore_blocks(student_blocks, rollback_state["blocks"])
                restored_parameters = named_trainable_parameters(student_blocks)
                strategy.load_state_dict(
                    rollback_state["strategy"],
                    restored_parameters,
                )
                t2t_history = [rollback_state]
                defense_diagnostics.update(
                    {
                        "rollback_to_task": int(rollback_state["task"]),
                        "rejected_task_indices": rejected_tasks,
                        "provisional_current_after": provisional_current,
                        "provisional_old_after": provisional_old,
                    }
                )
                for previous_row in performance["tasks"]:
                    if previous_row["task"] == rejected_tasks[0]:
                        previous_row["rejected_later_by_t2t_at_task"] = task_index
                current_importance = importance_summary(
                    getattr(strategy, "importance", None)
                )
            else:
                accepted_state = {
                    "task": task_index,
                    "parameters": current_parameter_snapshot,
                    "blocks": snapshot_blocks(student_blocks),
                    "strategy": strategy.state_dict(),
                    "hessian": current_hessian,
                    "regularizer": t2t_regularizer,
                }
                t2t_history.append(accepted_state)
                t2t_history = t2t_history[-2:]

        after = metric_view(
            evaluate(student_blocks, loader_eval, args, max_batches=args.eval_max_batches)
        )
        old = metric_view(
            evaluate(student_blocks, old_loader, args, max_batches=args.eval_max_batches)
        )
        performance["stability"]["acc"].append(old["acc"])
        performance["stability"]["mf1"].append(old["mf1"])
        seen_subjects.append(int(subject))

        if attack_diagnostics is not None:
            attack_diagnostics["materialized_files_retained"] = not (
                args.delete_poisoned_inputs_after_task
            )
        if args.delete_poisoned_inputs_after_task:
            deleted_files = delete_generated_inputs(
                generated_poisoned_paths,
                method_dir / "poisoned_inputs",
            )
            if attack_diagnostics is not None:
                attack_diagnostics["materialized_files_deleted"] = deleted_files

        task_row = {
            "task": task_index,
            "subject": int(subject),
            "current_before": before,
            "current_after": after,
            "old_generalization_after": old,
            "pseudo_labels": pseudo_diagnostics,
            "pseudo_labels_on_clean_current": clean_pseudo_diagnostics,
            "guiding_cpc_losses": cpc_losses,
            "noise": noise_diagnostics,
            "training": train_diagnostics,
            "importance": current_importance,
            "defense": defense_diagnostics,
            "progressive_proxy": progressive_task_row,
        }
        if args.noise_upload_root is None and args.n2n_manifest is None:
            task_row["attack"] = attack_diagnostics
            task_row["attack_label_guiding_cpc_losses"] = attack_label_cpc_losses
        performance["tasks"].append(task_row)

        if task_index in args.retention_milestones or task_index == total_tasks:
            performance["retention_snapshots"][str(task_index)] = (
                evaluate_seen_subjects(student_blocks, seen_subjects, args)
            )
        if not args.no_save_checkpoints and (
            task_index in args.checkpoint_milestones or task_index == total_tasks
        ):
            checkpoint_dir = method_dir / "checkpoints" / f"individual_{task_index}"
            save_blocks(student_blocks, checkpoint_dir, args.seed)
            torch.save(strategy.state_dict(), checkpoint_dir / "regularizer_state.pt")
            if robust_feature_defense is not None:
                torch.save(
                    robust_feature_defense.state_dict(),
                    checkpoint_dir / "icml2026_robust_feature_state.pt",
                )
            if t2t_detector is not None:
                torch.save(
                    t2t_detector.state_dict(),
                    checkpoint_dir / "icml2026_t2t_detector_state.pt",
                )

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
    if progressive_proxy is not None:
        performance["final"]["progressive_proxy"] = progressive_proxy.summary()
    summary = summarize_run(performance)
    performance["summary"] = summary
    save_progress(method_dir, performance)
    write_report(method_dir / "report.md", method, performance, summary)
    if progressive_proxy is not None:
        progressive_proxy.cleanup()
    return performance, summary


def vars_for_json(args) -> dict:
    output = {}
    for key, value in vars(args).items():
        if args.noise_upload_root is not None and key.startswith("attack_"):
            continue
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
    order_name = "seed_random"
    if getattr(args, "subject_order_manifest", None) is not None:
        manifest = json.loads(args.subject_order_manifest.read_text())
        if manifest.get("dataset") not in (None, args.dataset):
            raise ValueError("Subject-order manifest dataset mismatch")
        if int(manifest.get("partition_seed", args.seed)) != int(args.seed):
            raise ValueError("Subject-order manifest partition seed mismatch")
        manifest_order = [int(subject) for subject in manifest["new_order"]]
        if len(manifest_order) != len(set(manifest_order)):
            raise ValueError("Subject-order manifest contains duplicate subjects")
        if set(manifest_order) != set(new_order):
            raise ValueError("Subject-order manifest must permute the fixed incremental partition")
        new_order = manifest_order
        order_name = str(manifest.get("order_name", args.subject_order_manifest.stem))
    if args.max_subjects > 0:
        new_order = new_order[: args.max_subjects]
    return {
        "dataset": args.dataset,
        "seed": args.seed,
        "train_idx": [int(subject) for subject in train_idx],
        "val_idx": [int(subject) for subject in val_idx],
        "old_idx": [int(subject) for subject in old_idx],
        "new_order": new_order,
        "full_new_order": [int(subject) for subject in new_idx],
        "order_name": order_name,
        "subject_order_manifest": (
            None
            if getattr(args, "subject_order_manifest", None) is None
            else str(args.subject_order_manifest)
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ISRUC", "FACED"), default="ISRUC")
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
    parser.add_argument("--pretrain-seed", type=int, default=None)
    parser.add_argument("--subject-order-manifest", type=Path, default=None)
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
    parser.add_argument(
        "--defense-mode",
        choices=DEFENSE_MODES,
        default="none",
        help=(
            "ICML 2026 defense: task-to-task verification or the robust "
            "feature-space regularizer."
        ),
    )
    parser.add_argument(
        "--t2t-param-scope",
        choices=("all", "head", "classifier"),
        default="all",
    )
    parser.add_argument(
        "--t2t-action",
        choices=("rollback", "monitor"),
        default="rollback",
        help=(
            "Use monitor while calibrating a clean score distribution; "
            "rollback reproduces Algorithm 1 after a threshold crossing."
        ),
    )
    parser.add_argument("--t2t-threshold-multiplier", type=float, default=2.5)
    parser.add_argument("--t2t-window", type=int, default=5)
    parser.add_argument("--t2t-minimum-history", type=int, default=1)
    parser.add_argument("--t2t-pinv-rtol", type=float, default=1e-6)
    parser.add_argument("--t2t-eps", type=float, default=1e-12)
    parser.add_argument("--t2t-score-floor", type=float, default=1e-12)
    parser.add_argument("--robust-feature-sigma2", type=float, default=1.0)
    parser.add_argument(
        "--robust-feature-budget-per-dimension",
        type=float,
        default=2000.0 / (768.0 * 100.0),
        help=(
            "Assumed bounded non-shifted attack budget per classifier "
            "parameter; the default transfers the paper's CIFAR-100 ratio."
        ),
    )
    parser.add_argument(
        "--robust-feature-initial-risk",
        type=float,
        default=0.0,
        help="Per-direction R_0; zero estimates it from classifier weight energy.",
    )
    parser.add_argument("--robust-feature-eps", type=float, default=1e-12)
    parser.add_argument(
        "--robust-feature-max-regularizer",
        type=float,
        default=1e6,
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
    parser.add_argument(
        "--attack-generation-batch",
        type=int,
        default=1,
        help="Sequences jointly optimized by proxy_dual_harm per white-box step.",
    )
    parser.add_argument(
        "--attack-max-relative-l2",
        type=float,
        default=0.0,
        help="Optional per-sequence relative L2 cap in addition to attack-eps-scale.",
    )
    parser.add_argument("--attack-target-weight", type=float, default=1.0)
    parser.add_argument("--attack-conflict-weight", type=float, default=1.0)
    parser.add_argument("--attack-gradient-norm-weight", type=float, default=0.1)
    parser.add_argument("--attack-virtual-old-weight", type=float, default=1.0)
    parser.add_argument("--attack-virtual-new-weight", type=float, default=1.0)
    parser.add_argument("--attack-new-proxy-weight", type=float, default=1.0)
    parser.add_argument("--attack-curvature-scale", type=float, default=1.0)
    parser.add_argument("--attack-min-confidence", type=float, default=0.0)
    parser.add_argument("--attack-confidence-weight", type=float, default=0.0)
    parser.add_argument("--attack-l2-weight", type=float, default=0.0)
    parser.add_argument(
        "--attack-proxy-repeat",
        type=int,
        default=0,
        help=(
            "Extra data-only upload copies per poisoned sequence. This changes "
            "the incoming stream length but never learner labels or losses."
        ),
    )
    parser.add_argument(
        "--noise-upload-root",
        type=Path,
        default=None,
        help="Read a fixed per-task proxy-noise stream while keeping clean labels.",
    )
    parser.add_argument(
        "--n2n-manifest",
        type=Path,
        default=None,
        help=(
            "Canonical partial N-to-N signal-replacement manifest. Selected "
            "EEG/EOG paths change while clean annotation paths stay evaluator-only."
        ),
    )
    parser.add_argument(
        "--n2n-verify",
        choices=("none", "selected", "full"),
        default="selected",
        help="Payload verification performed when resolving each manifest task.",
    )
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument(
        "--delete-poisoned-inputs-after-task",
        action="store_true",
        help=(
            "Delete only runner-generated poisoned .npy files after the task's "
            "training, importance estimation, and evaluation are complete."
        ),
    )
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--retention-milestones", type=str, default="10,25,49")
    parser.add_argument("--checkpoint-milestones", type=str, default="0,1,10,25,49")
    add_progressive_proxy_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    requested_methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(requested_methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if not requested_methods:
        raise ValueError("At least one method is required")
    if args.defense_mode == "t2t" and "finetune" in requested_methods:
        raise ValueError(
            "T2T requires a non-zero historical quadratic regularizer; "
            "Finetune has no H_t. Use EWC, Online EWC, SI, or MAS."
        )
    if args.t2t_window < 1 or args.t2t_minimum_history < 1:
        raise ValueError("T2T window and minimum history must be positive")
    if args.t2t_minimum_history > args.t2t_window:
        raise ValueError("T2T minimum history cannot exceed its window")
    if args.t2t_threshold_multiplier <= 0:
        raise ValueError("T2T threshold multiplier must be positive")
    if args.t2t_pinv_rtol < 0 or args.t2t_eps <= 0:
        raise ValueError("T2T tolerances must be non-negative and positive")
    if args.robust_feature_sigma2 <= 0:
        raise ValueError("Robust-feature sigma2 must be positive")
    if args.robust_feature_budget_per_dimension < 0:
        raise ValueError("Robust-feature budget must be non-negative")
    if args.robust_feature_initial_risk < 0:
        raise ValueError("Robust-feature initial risk cannot be negative")
    if args.robust_feature_eps <= 0 or args.robust_feature_max_regularizer <= 0:
        raise ValueError("Robust-feature numerical limits must be positive")

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
    validate_progressive_proxy_args(args, len(split["new_order"]))
    args.attack_tasks = resolve_attack_tasks(
        args.attack_tasks,
        len(split["new_order"]),
    )
    if args.attack_mode == "none" and args.attack_tasks:
        raise ValueError("--attack-tasks requires a non-'none' --attack-mode")
    if args.attack_mode != "none" and not args.attack_tasks:
        raise ValueError("A non-'none' --attack-mode requires --attack-tasks")
    if args.n2n_manifest is not None:
        args.n2n_manifest = args.n2n_manifest.resolve()
        if args.attack_mode != "none" or args.attack_tasks:
            raise ValueError("Canonical N-to-N input cannot be combined with attack modes")
        if args.noise_upload_root is not None:
            raise ValueError("Use only one of --n2n-manifest and --noise-upload-root")
        if not args.n2n_manifest.is_file():
            raise FileNotFoundError(
                f"Canonical N-to-N manifest does not exist: {args.n2n_manifest}"
            )
    if args.noise_upload_root is not None:
        if args.attack_mode != "none" or args.attack_tasks:
            raise ValueError("External proxy noise cannot be combined with attack modes")
        if not args.noise_upload_root.is_dir():
            raise FileNotFoundError(
                f"Proxy-noise upload root does not exist: {args.noise_upload_root}"
            )
    if args.progressive_proxy_mode != "none" and (
        args.attack_mode != "none"
        or args.attack_tasks
        or args.n2n_manifest is not None
        or args.noise_upload_root is not None
    ):
        raise ValueError(
            "Progressive feedback proxy cannot be combined with another input mode"
        )
    if not 0.0 <= args.attack_fraction <= 1.0:
        raise ValueError("--attack-fraction must be in [0, 1]")
    if args.attack_steps < 1:
        raise ValueError("--attack-steps must be positive")
    if args.attack_reference_batch < 1:
        raise ValueError("--attack-reference-batch must be positive")
    if args.attack_generation_batch < 1:
        raise ValueError("--attack-generation-batch must be positive")
    if args.attack_max_relative_l2 < 0:
        raise ValueError("--attack-max-relative-l2 cannot be negative")
    if args.attack_proxy_repeat < 0:
        raise ValueError("--attack-proxy-repeat cannot be negative")
    if not 0.0 <= args.attack_min_confidence <= 1.0:
        raise ValueError("--attack-min-confidence must be in [0, 1]")
    if args.attack_mode == "proxy_dual_harm" and args.attack_param_scope != "classifier":
        raise ValueError("proxy_dual_harm currently requires --attack-param-scope classifier")
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
