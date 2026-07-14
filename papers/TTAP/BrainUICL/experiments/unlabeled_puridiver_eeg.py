"""Unlabeled target-subject CL using guiding pseudo labels and PuriDivER.

The source-supervised compact guide can remain frozen or cumulatively adapt its
encoder with a label-free CPC objective for each incoming subject. The guide's
classifier stays frozen so that its outputs retain sleep-stage semantics. The
student can copy the source guide or start from independent random weights.

Every target adaptation epoch receives an argmax pseudo label; there is no
confidence gate. The student uses PuriDivER memory construction and robust
replay without BrainUICL CEA, teacher entropy/disagreement, confidence-based
selection, source-label protection, or BrainUICL's joint-update objective. The
optional CPC-style guide adaptation is recorded explicitly in result metadata.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from pure_puridiver_eeg import (
    NUM_CLASSES,
    CompactEEGClassifier,
    EpochPool,
    PoolDataset,
    PuriDivERMemory,
    discover_subjects,
    evaluate,
    fix_seed,
    infer_pool,
    load_epoch_pool,
    make_optimizer,
    normalize_epochs,
    online_train_subject,
    reference_subject_split,
    replay_train,
    save_metrics,
    serializable_args,
    split_subject_paths,
    summarize_matrix,
)


class SequencePathDataset(Dataset):
    def __init__(self, data_root: Path, subjects: list[int]):
        self.records: list[tuple[Path, Path]] = []
        for subject in subjects:
            data_directory = data_root / str(subject) / "data"
            label_directory = data_root / str(subject) / "label"
            paths = sorted(data_directory.glob("*.npy"), key=lambda path: int(path.stem))
            self.records.extend((path, label_directory / path.name) for path in paths)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        data_path, label_path = self.records[index]
        x = torch.from_numpy(normalize_epochs(np.load(data_path))).float()
        y = torch.from_numpy(np.load(label_path).astype(np.int64))
        return x, y


class UnlabeledSequencePathDataset(Dataset):
    """Target sequence loader that never opens annotation files."""

    def __init__(self, paths: list[Path]):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return torch.from_numpy(normalize_epochs(np.load(self.paths[index]))).float()


class TemporalCPCHeads(nn.Module):
    def __init__(self, feature_dim: int, prediction_steps: int):
        super().__init__()
        self.context = nn.GRU(feature_dim, feature_dim, batch_first=True)
        self.predictors = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim) for _ in range(prediction_steps)]
        )


def initialize_student(guiding_model: CompactEEGClassifier, args) -> CompactEEGClassifier:
    if args.student_initialization == "guide_copy":
        student = CompactEEGClassifier().to(args.device)
        student.load_state_dict(guiding_model.state_dict())
        return student
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 17)
        return CompactEEGClassifier().to(args.device)


def adapt_guiding_model_cpc(
    guiding_model: CompactEEGClassifier,
    adaptation_paths: list[Path],
    args,
    task_index: int,
) -> dict:
    if args.guide_policy == "frozen":
        return {
            "policy": "frozen",
            "epochs": 0,
            "updates": 0,
            "mean_loss": None,
        }

    dataset = UnlabeledSequencePathDataset(adaptation_paths)
    if len(dataset) < 2:
        raise RuntimeError("CPC guide adaptation requires at least two target sequences")
    batch_size = min(args.guide_cpc_sequence_batch, len(dataset))
    generator = torch.Generator().manual_seed(args.seed + 1_000_000 + task_index)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        drop_last=len(dataset) > batch_size,
        generator=generator,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 2_000_000 + task_index)
        heads = TemporalCPCHeads(128, args.guide_cpc_prediction_steps).to(args.device)

    for parameter in guiding_model.parameters():
        parameter.requires_grad_(False)
    for parameter in guiding_model.encoder.parameters():
        parameter.requires_grad_(True)
    classifier_before = {
        key: value.detach().clone() for key, value in guiding_model.classifier.state_dict().items()
    }
    optimizer = torch.optim.Adam(
        list(guiding_model.encoder.parameters()) + list(heads.parameters()),
        lr=args.guide_cpc_learning_rate,
        weight_decay=1e-4,
    )
    losses = []
    for _epoch in range(args.guide_cpc_epochs):
        guiding_model.encoder.train()
        heads.train()
        for sequence in loader:
            sequence = sequence.to(args.device)
            batch, sequence_length = sequence.shape[:2]
            encoded = guiding_model.encoder(sequence.reshape(-1, 8, 3000)).squeeze(-1)
            encoded = encoded.reshape(batch, sequence_length, -1)
            max_start = sequence_length - args.guide_cpc_prediction_steps - 1
            min_start = min(5, max_start)
            start = int(
                torch.randint(
                    min_start,
                    max_start + 1,
                    (1,),
                    device=args.device,
                ).item()
            )
            context_sequence = encoded[:, : start + 1]
            context_output, _hidden = heads.context(context_sequence)
            context = context_output[:, -1]
            labels = torch.arange(batch, device=args.device)
            loss = torch.zeros((), device=args.device)
            for step, predictor in enumerate(heads.predictors, start=1):
                prediction = F.normalize(predictor(context), dim=1)
                target = F.normalize(encoded[:, start + step], dim=1)
                logits = prediction @ target.T / args.guide_cpc_temperature
                loss = loss + F.cross_entropy(logits, labels)
            loss = loss / len(heads.predictors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    for key, value in guiding_model.classifier.state_dict().items():
        if not torch.equal(value, classifier_before[key]):
            raise RuntimeError("CPC unexpectedly modified the guiding classifier")
    for parameter in guiding_model.parameters():
        parameter.requires_grad_(False)
    guiding_model.eval()
    return {
        "policy": "cpc_dynamic",
        "epochs": args.guide_cpc_epochs,
        "updates": len(losses),
        "mean_loss": float(np.mean(losses)),
    }


@torch.no_grad()
def evaluate_sequence_dataset(model, dataset: SequencePathDataset, args) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=args.source_sequence_batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    predictions, labels = [], []
    model.eval()
    for x, y in loader:
        x = x.reshape(-1, 8, 3000).to(args.device)
        y = y.reshape(-1)
        predictions.append(model(x).argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    prediction = np.concatenate(predictions)
    label = np.concatenate(labels)
    return {
        "acc": float(accuracy_score(label, prediction)),
        "mf1": float(
            f1_score(label, prediction, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
        ),
        "epochs": int(label.size),
    }


def train_or_load_guiding_model(args, train_subjects: list[int], val_subjects: list[int]):
    model = CompactEEGClassifier().to(args.device)
    checkpoint_path = args.guide_checkpoint
    metadata_path = checkpoint_path.with_suffix(".json")
    if checkpoint_path.exists() and not args.retrain_guide:
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        metadata = checkpoint.get("metadata", {})
        metadata["loaded_from_cache"] = True
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, metadata

    train_dataset = SequencePathDataset(args.data_root, train_subjects)
    val_dataset = SequencePathDataset(args.data_root, val_subjects)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.source_sequence_batch,
        shuffle=True,
        num_workers=args.num_worker,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.guide_learning_rate,
        weight_decay=1e-4,
    )
    best_state = None
    best_mf1 = -1.0
    history = []
    for epoch in range(args.guide_epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.reshape(-1, 8, 3000).to(args.device)
            y = y.reshape(-1).to(args.device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate_sequence_dataset(model, val_dataset, args)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation": validation}
        history.append(row)
        print(
            f"[guide] epoch={epoch + 1}/{args.guide_epochs} loss={row['loss']:.4f} "
            f"val_acc={validation['acc']:.4f} val_mf1={validation['mf1']:.4f}",
            flush=True,
        )
        if validation["mf1"] > best_mf1:
            best_mf1 = validation["mf1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Guiding model training produced no checkpoint")
    model.load_state_dict(best_state)
    final_validation = evaluate_sequence_dataset(model, val_dataset, args)
    metadata = {
        "type": "source_supervised_compact_eeg_guide",
        "source_train_subjects": train_subjects,
        "source_val_subjects": val_subjects,
        "guide_epochs": args.guide_epochs,
        "best_validation_mf1": best_mf1,
        "loaded_best_validation": final_validation,
        "history": history,
        "loaded_from_cache": False,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "metadata": metadata}, checkpoint_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata


@torch.no_grad()
def assign_all_guiding_pseudo_labels(
    guiding_model: CompactEEGClassifier,
    unlabeled_pool: EpochPool,
    args,
) -> tuple[EpochPool, dict]:
    logits, _features = infer_pool(
        guiding_model,
        unlabeled_pool,
        args.device,
        args.infer_batch_size,
    )
    probabilities = logits.softmax(dim=1)
    confidence, pseudo_labels = probabilities.max(dim=1)
    true_labels = unlabeled_pool.true_y
    pseudo_pool = EpochPool(
        x=unlabeled_pool.x,
        observed_y=pseudo_labels.long(),
        true_y=true_labels,
        subject_y=unlabeled_pool.subject_y,
    )
    diagnostics = {
        "candidate_rule": "all_guiding_argmax_epochs",
        "confidence_threshold": None,
        "candidate_epochs": len(unlabeled_pool),
        "accepted_epochs": len(unlabeled_pool),
        "acceptance_rate": 1.0,
        "mean_confidence_diagnostic_only": float(confidence.mean()),
        "pseudo_label_acc_diagnostic_only": float(pseudo_labels.eq(true_labels).float().mean()),
        "pseudo_label_mf1_diagnostic_only": float(
            f1_score(
                true_labels.numpy(),
                pseudo_labels.numpy(),
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "pseudo_class_counts": torch.bincount(pseudo_labels, minlength=NUM_CLASSES).tolist(),
    }
    return pseudo_pool, diagnostics


def brainuicl_stability_summary(stability: dict[str, list[float]]) -> dict:
    initial_acc = float(stability["ACC"][0])
    initial_mf1 = float(stability["MF1"][0])
    final_acc = float(stability["ACC"][-1])
    final_mf1 = float(stability["MF1"][-1])
    old_acc_change = final_acc - initial_acc
    return {
        "initial_acc": initial_acc,
        "initial_mf1": initial_mf1,
        "acc": final_acc,
        "mf1": final_mf1,
        "aaa": float(np.mean(stability["ACC"])),
        "aaf1": float(np.mean(stability["MF1"])),
        # Kept for exact compatibility with BrainUICL's published metric. It is
        # an absolute relative endpoint change, so a value > 1 can describe
        # improvement from a weak random initialization rather than forgetting.
        "fr": float(abs(initial_acc - final_acc) / initial_acc),
        "old_acc_change": old_acc_change,
        "old_mf1_change": final_mf1 - initial_mf1,
        "relative_old_acc_change": old_acc_change / initial_acc,
    }


def brainuicl_plasticity_summary(plasticity: dict[str, dict[str, list[float]]]) -> dict:
    acc = np.asarray([row["ACC"] for row in plasticity.values()], dtype=np.float64)
    mf1 = np.asarray([row["MF1"] for row in plasticity.values()], dtype=np.float64)
    return {
        "initial_acc": float(acc[:, 0].mean()),
        "before_acc": float(acc[:, 1].mean()),
        "after_acc": float(acc[:, 2].mean()),
        "initial_mf1": float(mf1[:, 0].mean()),
        "before_mf1": float(mf1[:, 1].mean()),
        "after_mf1": float(mf1[:, 2].mean()),
    }


def update_brainuicl_stability(stability: dict[str, list[float]], result: dict) -> None:
    stability["ACC"].append(float(result["acc"]))
    stability["MF1"].append(float(result["mf1"]))
    stability["AAA"].append(float(np.mean(stability["ACC"])))
    stability["AAF1"].append(float(np.mean(stability["MF1"])))
    initial_acc = stability["ACC"][0]
    stability["FR"].append(float(abs(initial_acc - stability["ACC"][-1]) / initial_acc))


def run(args) -> dict:
    fix_seed(args.seed)
    subjects = discover_subjects(args.data_root)
    split = reference_subject_split(subjects, args.seed)
    order = split["new_order"]
    if args.max_subjects > 0:
        order = order[: args.max_subjects]

    guiding_model, guide_metadata = train_or_load_guiding_model(
        args,
        split["train"],
        split["val"],
    )
    student = initialize_student(guiding_model, args)
    initial_student = copy.deepcopy(student).to(args.device).eval()
    for parameter in initial_student.parameters():
        parameter.requires_grad_(False)
    memory = PuriDivERMemory(args.memory_size, args.seed)
    test_pools: list[EpochPool] = []
    acc_matrix = np.full((len(order), len(order)), np.nan, dtype=np.float64)
    mf1_matrix = np.full_like(acc_matrix, np.nan)
    task_rows = []
    payload = {
        "config": serializable_args(args),
        "split": split,
        "order": order,
        "protocol": {
            "target_supervision": "none",
            "target_annotation_used_for_training": False,
            "target_annotation_used_for_selection": False,
            "target_annotation_use": (
                "full_subject_evaluation_and_posthoc_diagnostics_only"
                if args.evaluation_protocol == "brainuicl"
                else "held_out_evaluation_and_posthoc_diagnostics_only"
            ),
            "candidate_rule": "all_argmax_pseudo_labels_without_confidence_filter",
            "brainuicl_components": [],
            "target_cpc_guide_adaptation": args.guide_policy == "cpc_dynamic",
            "guide_parameter_updates": (
                "encoder_only_classifier_frozen"
                if args.guide_policy == "cpc_dynamic"
                else "none"
            ),
            "guiding_policy": args.guide_policy,
            "student_initialization": args.student_initialization,
            "evaluation_protocol": args.evaluation_protocol,
        },
        "guide": guide_metadata,
        "tasks": task_rows,
    }
    brainuicl_performance = None
    old_dataset = None
    if args.evaluation_protocol == "brainuicl":
        old_dataset = SequencePathDataset(args.data_root, split["old_generalization"])
        initial_old = evaluate_sequence_dataset(student, old_dataset, args)
        brainuicl_performance = {
            "stability": {"ACC": [], "MF1": [], "AAA": [], "AAF1": [], "FR": []},
            "plasticity": {
                str(subject): {"ACC": [], "MF1": []} for subject in order
            },
            "old_generalization_subjects": split["old_generalization"],
            "new_order": order,
        }
        update_brainuicl_stability(brainuicl_performance["stability"], initial_old)
        payload["brainuicl_performance"] = brainuicl_performance
    save_metrics(args.output_root, payload)

    for task_index, subject in enumerate(order):
        if args.evaluation_protocol == "brainuicl":
            data_directory = args.data_root / str(subject) / "data"
            adaptation_paths = sorted(
                data_directory.glob("*.npy"), key=lambda path: int(path.stem)
            )
            test_paths = adaptation_paths
            new_dataset = SequencePathDataset(args.data_root, [subject])
        else:
            adaptation_paths, test_paths = split_subject_paths(
                args.data_root, subject, args.train_fraction
            )
            new_dataset = None
        guide_adaptation = adapt_guiding_model_cpc(
            guiding_model,
            adaptation_paths,
            args,
            task_index + 1,
        )
        diagnostic_pool, adaptation_stats = load_epoch_pool(
            args.data_root,
            subject,
            adaptation_paths,
            noise_rate=0.0,
            noise_seed=args.seed,
        )
        pseudo_pool, pseudo_diagnostics = assign_all_guiding_pseudo_labels(
            guiding_model, diagnostic_pool, args
        )
        if args.evaluation_protocol == "brainuicl":
            test_pool = None
            test_stats = {
                "epochs": len(diagnostic_pool),
                "sequences": len(test_paths),
                "scope": "full_subject_same_as_unlabeled_adaptation",
            }
            initial = evaluate_sequence_dataset(initial_student, new_dataset, args)
            before = evaluate_sequence_dataset(student, new_dataset, args)
        else:
            test_pool, test_stats = load_epoch_pool(
                args.data_root,
                subject,
                test_paths,
                noise_rate=0.0,
                noise_seed=args.seed,
            )
            initial = None
            before = evaluate(student, test_pool, args)
        if args.method == "guide_only":
            online = {
                "mini_batches": 0,
                "mean_online_loss": None,
                "mean_diversity_coefficient": None,
                "last_memory_update": None,
            }
            replay = []
        else:
            optimizer = make_optimizer(student, args.learning_rate)
            online = online_train_subject(
                student,
                optimizer,
                pseudo_pool,
                memory,
                args,
                task_index + 1,
            )
            replay = replay_train(student, optimizer, memory, args, task_index + 1)
        if args.evaluation_protocol == "brainuicl":
            after = evaluate_sequence_dataset(student, new_dataset, args)
            old_result = evaluate_sequence_dataset(student, old_dataset, args)
            update_brainuicl_stability(brainuicl_performance["stability"], old_result)
            brainuicl_performance["plasticity"][str(subject)] = {
                "ACC": [initial["acc"], before["acc"], after["acc"]],
                "MF1": [initial["mf1"], before["mf1"], after["mf1"]],
            }
            old_acc = old_result["acc"]
            old_mf1 = old_result["mf1"]
            all_seen_acc = None
            all_seen_mf1 = None
        else:
            test_pools.append(test_pool)
            for seen_index, seen_pool in enumerate(test_pools):
                result = evaluate(student, seen_pool, args)
                acc_matrix[task_index, seen_index] = result["acc"]
                mf1_matrix[task_index, seen_index] = result["mf1"]
            after = {
                "acc": float(acc_matrix[task_index, task_index]),
                "mf1": float(mf1_matrix[task_index, task_index]),
                "epochs": len(test_pool),
            }
            old_acc = float(np.nanmean(acc_matrix[task_index, :task_index])) if task_index else None
            old_mf1 = float(np.nanmean(mf1_matrix[task_index, :task_index])) if task_index else None
            all_seen_acc = float(np.nanmean(acc_matrix[task_index, : task_index + 1]))
            all_seen_mf1 = float(np.nanmean(mf1_matrix[task_index, : task_index + 1]))
        row = {
            "task": task_index + 1,
            "subject": subject,
            "adaptation": adaptation_stats,
            "test": test_stats,
            "pseudo_labels": pseudo_diagnostics,
            "guide_adaptation": guide_adaptation,
            "mini_batches_expected": int(math.ceil(len(pseudo_pool) / args.online_batch_size)),
            "initial": initial,
            "before": before,
            "after": after,
            "plasticity": {
                "acc_gain": after["acc"] - before["acc"],
                "mf1_gain": after["mf1"] - before["mf1"],
            },
            "old_subject_mean_after": {"acc": old_acc, "mf1": old_mf1},
            "all_seen_mean_after": {
                "acc": all_seen_acc,
                "mf1": all_seen_mf1,
            },
            "online": online,
            "replay": replay,
            "memory": {
                "size": len(memory),
                "pseudo_label_purity_diagnostic_only": (
                    float(memory.pool.observed_y.eq(memory.pool.true_y).float().mean())
                    if len(memory)
                    else None
                ),
                "pseudo_class_counts": torch.bincount(
                    memory.pool.observed_y, minlength=NUM_CLASSES
                ).tolist(),
            },
        }
        task_rows.append(row)
        common_summary = {
            "mean_new_subject_acc_gain": float(
                np.mean([task["plasticity"]["acc_gain"] for task in task_rows])
            ),
            "mean_new_subject_mf1_gain": float(
                np.mean([task["plasticity"]["mf1_gain"] for task in task_rows])
            ),
            "mean_guiding_pseudo_label_acc": float(
                np.mean(
                    [task["pseudo_labels"]["pseudo_label_acc_diagnostic_only"] for task in task_rows]
                )
            ),
            "mean_guiding_pseudo_label_mf1": float(
                np.mean(
                    [task["pseudo_labels"]["pseudo_label_mf1_diagnostic_only"] for task in task_rows]
                )
            ),
        }
        if args.evaluation_protocol == "brainuicl":
            payload["summary"] = {
                **brainuicl_stability_summary(brainuicl_performance["stability"]),
                "plasticity": brainuicl_plasticity_summary(
                    {
                        key: value
                        for key, value in brainuicl_performance["plasticity"].items()
                        if value["ACC"]
                    }
                ),
                **common_summary,
            }
        else:
            payload["acc_matrix"] = acc_matrix.tolist()
            payload["mf1_matrix"] = mf1_matrix.tolist()
            payload["summary"] = {
                "acc": summarize_matrix(acc_matrix[: task_index + 1, : task_index + 1]),
                "mf1": summarize_matrix(mf1_matrix[: task_index + 1, : task_index + 1]),
                **common_summary,
            }
        save_metrics(args.output_root, payload)
        purity = row["memory"]["pseudo_label_purity_diagnostic_only"]
        print(
            f"[unlabeled-{args.method}] task={task_index + 1}/{len(order)} subject={subject} "
            f"pseudo_acc={pseudo_diagnostics['pseudo_label_acc_diagnostic_only']:.4f} "
            f"new_acc={before['acc']:.4f}->{after['acc']:.4f} "
            f"old_acc={old_acc if old_acc is not None else float('nan'):.4f} "
            f"seen_acc={all_seen_acc if all_seen_acc is not None else float('nan'):.4f} "
            f"memory={len(memory)} "
            f"purity={purity if purity is not None else float('nan'):.4f}",
            flush=True,
        )

    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent
        / "rttdp_brainuicl_runs"
        / "unlabeled_puridiver_eeg",
    )
    parser.add_argument(
        "--guide-checkpoint",
        type=Path,
        default=Path(
            "/home/undefined/Disk/ai-storage/BrainUICL/model_parameter/"
            "PseudoPuriDivER/compact_guide_seed4321.pt"
        ),
    )
    parser.add_argument("--retrain-guide", action="store_true")
    parser.add_argument(
        "--guide-policy",
        choices=["frozen", "cpc_dynamic"],
        default="frozen",
    )
    parser.add_argument(
        "--student-initialization",
        choices=["guide_copy", "random"],
        default="guide_copy",
    )
    parser.add_argument(
        "--method",
        choices=["puridiver", "puri_memory", "er", "guide_only"],
        default="puridiver",
    )
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument(
        "--evaluation-protocol",
        choices=["heldout_matrix", "brainuicl"],
        default="heldout_matrix",
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--guide-epochs", type=int, default=15)
    parser.add_argument("--guide-learning-rate", type=float, default=1e-3)
    parser.add_argument("--guide-cpc-epochs", type=int, default=3)
    parser.add_argument("--guide-cpc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--guide-cpc-sequence-batch", type=int, default=8)
    parser.add_argument("--guide-cpc-prediction-steps", type=int, default=3)
    parser.add_argument("--guide-cpc-temperature", type=float, default=0.1)
    parser.add_argument("--source-sequence-batch", type=int, default=4)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--memory-size", type=int, default=1000)
    parser.add_argument("--online-batch-size", type=int, default=128)
    parser.add_argument("--replay-batch-size", type=int, default=128)
    parser.add_argument("--infer-batch-size", type=int, default=256)
    parser.add_argument("--replay-epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-diversity-coefficient", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--weak-noise", type=float, default=0.01)
    parser.add_argument("--weak-scale", type=float, default=0.02)
    parser.add_argument("--strong-noise", type=float, default=0.05)
    parser.add_argument("--strong-scale", type=float, default=0.10)
    parser.add_argument("--strong-mask-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be in (0, 1)")
    args.device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    return args


if __name__ == "__main__":
    run(parse_args())
