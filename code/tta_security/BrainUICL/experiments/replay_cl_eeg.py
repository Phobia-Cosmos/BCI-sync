#!/usr/bin/env python3
"""Aligned replay methods on the unlabeled ISRUC continual-learning protocol.

This runner deliberately reuses the source-pretrained BrainUICL network as a
backbone only.  The continual-learning algorithm is plain reservoir ER:

* no BrainUICL confidence gate, CEA, DCB, source replay, or growing buffer;
* no EWC/SI/MAS parameter penalty;
* current targets are hard labels from the task-local CPC guide;
* previous sequences are sampled 1:1 from a fixed reservoir with their stored
  admission-time pseudo labels.

The runner also supports SPR-style filtered replay and two PuriDivER-style
variants.  Fixed uploads can be reused across methods so that the defense
comparison does not regenerate a victim-adaptive attack for each method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.regularization_cl import (  # noqa: E402
    freeze_batch_norm_running_stats,
    named_trainable_parameters,
)
from aligned_replay_defenses import (  # noqa: E402
    PuriDivERSequenceMemory,
    ReplayRecord,
    ReservoirReplayMemory,
    apply_spr_filter,
    build_cru_state,
    build_memory_records,
    collect_epoch_outputs,
    load_replay_batch,
    puridiver_branch_loss,
)
from regularization_cl_attacks import (  # noqa: E402
    materialize_batched_proxy_dual_harm_subject,
)
from regularization_cl_eeg import (  # noqa: E402
    adapt_guiding_model,
    build_split,
    evaluate_seen_subjects,
    metric_view,
    parse_int_set,
    pseudo_label_diagnostics,
    resolve_attack_tasks,
    summarize_run,
)
from rttdp_brainuicl_full import (  # noqa: E402
    SequenceDataset,
    evaluate,
    flat_logits,
    forward_blocks,
    load_pretrained,
    make_loader,
    set_train,
    subject_paths,
)
from utils.config import ModelConfig  # noqa: E402
from utils.util import fix_randomness  # noqa: E402


ATTACK_MODES = ("none", "benign_repeat", "proxy_dual_harm")
REPLAY_METHODS = (
    "plain_er",
    "spr_er",
    "puridiver_memory_ce",
    "puridiver_cru",
)
METHOD_DESCRIPTIONS = {
    "plain_er": "fixed-capacity reservoir experience replay",
    "spr_er": "reservoir experience replay with SPR epoch-level admission masks",
    "puridiver_memory_ce": "PuriDivER-style purity/diversity sequence memory with CE replay",
    "puridiver_cru": "PuriDivER-style purity/diversity memory with clean/relabel/unlabeled replay loss",
}


class UnlabeledSequenceDataset(Dataset):
    """Load target signals without opening target annotation files."""

    def __init__(self, data_paths: Sequence[Path], sequence_length: int):
        self.data_paths = [Path(path) for path in data_paths]
        self.sequence_length = int(sequence_length)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int):
        values = torch.from_numpy(
            np.load(self.data_paths[index]).astype(np.float32)
        )
        dummy = torch.zeros(self.sequence_length, dtype=torch.long)
        return values[:, :2, :], values[:, 2:, :], dummy


def make_unlabeled_loader(
    data_paths: Sequence[Path],
    args,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        UnlabeledSequenceDataset(data_paths, args.model_param.SeqLength),
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=args.num_worker,
    )


@torch.no_grad()
def infer_pseudo_labels(
    guiding_blocks,
    data_paths: Sequence[Path],
    args,
) -> list[np.ndarray]:
    loader = make_unlabeled_loader(data_paths, args, shuffle=False)
    set_train(guiding_blocks, False)
    rows: list[np.ndarray] = []
    for eog, eeg, _dummy in loader:
        logits = forward_blocks(
            guiding_blocks,
            eog.to(args.device),
            eeg.to(args.device),
            args,
        )
        labels = logits.argmax(dim=1).detach().cpu().numpy()
        rows.extend(labels[index].astype(np.int64) for index in range(len(labels)))
    if len(rows) != len(data_paths):
        raise RuntimeError("Pseudo-label inference did not preserve upload count")
    return rows


def train_er_task(
    student_blocks,
    guiding_blocks,
    current_loader: DataLoader,
    memory,
    args,
    task_index: int,
    subject: int,
) -> dict:
    parameters = named_trainable_parameters(student_blocks)
    optimizer = torch.optim.Adam(
        [parameter for _name, parameter in parameters],
        lr=args.cl_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    epoch_rows: list[dict] = []
    is_cru = args.method == "puridiver_cru"

    for epoch in range(1, args.incremental_epoch + 1):
        set_train(student_blocks, True)
        if args.freeze_bn_stats:
            freeze_batch_norm_running_stats(student_blocks)
        set_train(guiding_blocks, False)
        current_losses: list[float] = []
        replay_losses: list[float] = []
        total_losses: list[float] = []
        replay_sequences = 0
        poisoned_replay_sequences = 0
        cru_rows: list[dict] = []

        for batch_index, (eog, eeg, _dummy) in enumerate(current_loader):
            eog = eog.to(args.device)
            eeg = eeg.to(args.device)
            with torch.no_grad():
                current_labels = forward_blocks(
                    guiding_blocks, eog, eeg, args
                ).argmax(dim=1)
            current_logits = forward_blocks(student_blocks, eog, eeg, args)
            if is_cru:
                current_state = build_cru_state(
                    current_logits.detach(),
                    current_labels.detach(),
                    seed=args.seed + 1_000_000 * task_index + 10_000 * epoch + batch_index,
                    thresholds=(
                        args.puridiver_clean_threshold,
                        args.puridiver_uncertainty_threshold,
                    ),
                )
                current_loss, current_cru = puridiver_branch_loss(
                    student_blocks,
                    eog,
                    eeg,
                    current_labels,
                    current_state,
                    args,
                )
                cru_rows.append({"current": current_cru})
            else:
                current_loss = F.cross_entropy(
                    flat_logits(current_logits), current_labels.reshape(-1)
                )

            replay_records = memory.sample(eog.shape[0])
            if replay_records:
                replay_eog, replay_eeg, replay_labels = load_replay_batch(
                    replay_records
                )
                replay_eog = replay_eog.to(args.device)
                replay_eeg = replay_eeg.to(args.device)
                replay_labels = replay_labels.to(args.device)
                replay_logits = forward_blocks(
                    student_blocks, replay_eog, replay_eeg, args
                )
                if is_cru:
                    replay_state = build_cru_state(
                        replay_logits.detach(),
                        replay_labels.detach(),
                        seed=args.seed
                        + 2_000_000 * task_index
                        + 10_000 * epoch
                        + batch_index,
                        thresholds=(
                            args.puridiver_clean_threshold,
                            args.puridiver_uncertainty_threshold,
                        ),
                    )
                    replay_loss, replay_cru = puridiver_branch_loss(
                        student_blocks,
                        replay_eog,
                        replay_eeg,
                        replay_labels,
                        replay_state,
                        args,
                    )
                    cru_rows[-1]["replay"] = replay_cru
                else:
                    replay_loss = F.cross_entropy(
                        flat_logits(replay_logits),
                        replay_labels.reshape(-1),
                        ignore_index=-100,
                    )
                loss = 0.5 * (current_loss + replay_loss)
                replay_sequences += len(replay_records)
                poisoned_replay_sequences += sum(
                    int(record.poisoned) for record in replay_records
                )
            else:
                replay_loss = current_loss.new_zeros(())
                loss = current_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _name, parameter in parameters],
                    args.grad_clip,
                )
            optimizer.step()
            current_losses.append(float(current_loss.detach().cpu()))
            replay_losses.append(float(replay_loss.detach().cpu()))
            total_losses.append(float(loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "current_loss": float(np.mean(current_losses)),
            "replay_loss": float(np.mean(replay_losses)),
            "total_loss": float(np.mean(total_losses)),
            "replay_sequences": replay_sequences,
            "poisoned_replay_sequences": poisoned_replay_sequences,
            "poisoned_replay_fraction": (
                poisoned_replay_sequences / max(replay_sequences, 1)
            ),
        }
        if cru_rows:
            for branch in ("current", "replay"):
                values = [
                    row[branch]
                    for row in cru_rows
                    if branch in row
                ]
                if values:
                    row[f"{branch}_clean_count"] = int(
                        sum(item["clean_count"] for item in values)
                    )
                    row[f"{branch}_relabel_count"] = int(
                        sum(item["relabel_count"] for item in values)
                    )
                    row[f"{branch}_unlabeled_count"] = int(
                        sum(item["unlabeled_count"] for item in values)
                    )
        epoch_rows.append(row)
        print(
            f"[student:er] task={task_index} subject={subject} "
            f"epoch={epoch}/{args.incremental_epoch} "
            f"current={row['current_loss']:.6f} "
            f"replay={row['replay_loss']:.6f} "
            f"poison_replay={row['poisoned_replay_fraction']:.4f}",
            flush=True,
        )
    return {
        "epochs": epoch_rows,
        "last_current_loss": epoch_rows[-1]["current_loss"],
        "last_replay_loss": epoch_rows[-1]["replay_loss"],
        "last_total_loss": epoch_rows[-1]["total_loss"],
        "replay_sequences": sum(row["replay_sequences"] for row in epoch_rows),
        "poisoned_replay_sequences": sum(
            row["poisoned_replay_sequences"] for row in epoch_rows
        ),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def serializable_args(args) -> dict:
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


def make_memory(args):
    seed = args.seed + 700_001
    if args.method in {"plain_er", "spr_er"}:
        return ReservoirReplayMemory(args.memory_capacity, seed)
    return PuriDivERSequenceMemory(args.memory_capacity, seed)


def _fixed_attack_row(args, task_index: int) -> dict:
    if args.fixed_upload_root is None:
        raise ValueError("fixed upload root is required for a shared attack")
    metrics_path = args.fixed_upload_root / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Fixed-upload metrics are missing: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in payload.get("tasks", []):
        if int(row.get("task", -1)) == int(task_index):
            attack = row.get("attack") or {}
            if attack.get("mode") != "proxy_dual_harm":
                break
            return attack
    raise ValueError(f"No proxy-dual-harm source row for task {task_index}")


def _paths_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_fixed_attack_uploads(
    args,
    task_index: int,
    subject: int,
    clean_data_paths: Sequence[Path],
) -> tuple[list[Path], dict]:
    """Load immutable poisoned files generated by a separate source run."""

    if args.fixed_upload_root is None:
        raise ValueError("--fixed-upload-root is required for fixed shared uploads")
    directory = (
        args.fixed_upload_root
        / "poisoned_inputs"
        / f"task_{int(task_index)}_subject_{int(subject)}"
    )
    if not directory.is_dir():
        raise FileNotFoundError(f"Fixed-upload directory is missing: {directory}")
    poisoned_paths: list[Path] = []
    for clean_path in clean_data_paths:
        poisoned_path = directory / clean_path.name
        if not poisoned_path.is_file():
            raise FileNotFoundError(
                f"Fixed-upload sequence is missing for task {task_index}: {poisoned_path}"
            )
        poisoned_paths.append(poisoned_path)
    source_attack = _fixed_attack_row(args, task_index)
    diagnostics = dict(source_attack)
    diagnostics.update(
        {
            "fixed_shared_upload": True,
            "generated_inputs": False,
            "source_generation_run": str(args.fixed_upload_root),
            "source_generation_diagnostics_mean": source_attack.get(
                "diagnostics_mean", {}
            ),
            "fixed_upload_sha256": _paths_sha256(poisoned_paths),
            "poison_indices": list(range(len(clean_data_paths))),
            "poisoned_sequences": len(clean_data_paths),
            "total_sequences": len(clean_data_paths),
            "poison_fraction": 1.0,
        }
    )
    return poisoned_paths, diagnostics


def admit_memory(
    memory,
    records: list[ReplayRecord],
    data_paths: Sequence[Path],
    student_blocks,
    args,
) -> tuple[dict, dict | None, list[ReplayRecord]]:
    """Apply the selected label-free admission policy after task training."""

    if args.method == "spr_er":
        _logits, epoch_embeddings, _sequence_embeddings = collect_epoch_outputs(
            student_blocks, data_paths, args
        )
        retained, filter_stats = apply_spr_filter(
            records,
            epoch_embeddings,
            args,
            seed=args.seed + 3_000_000 + len(memory.records) + len(records),
        )
        update = memory.add(retained)
        update["spr"] = filter_stats
        return update, filter_stats, retained
    if args.method in {"puridiver_memory_ce", "puridiver_cru"}:
        update = memory.add(
            records,
            student_blocks,
            args,
            args.puridiver_diversity_coefficient,
        )
        return update, None, records
    update = memory.add(records)
    return update, None, records


def run(args) -> dict:
    fix_randomness(args.seed)
    split = build_split(args)
    args.attack_tasks = resolve_attack_tasks(
        args.attack_tasks, len(split["new_order"])
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(args.output_root / "split.json", split)
    save_json(args.output_root / "config.json", serializable_args(args))

    student_blocks = load_pretrained(args)
    initial_blocks = load_pretrained(args)
    memory = make_memory(args)
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
    source_reference_paths: list[Path] = []
    for source_subject in split["train_idx"]:
        source_reference_paths.extend(subject_paths(args.data_root, source_subject)[0])

    initial_old = metric_view(
        evaluate(student_blocks, old_loader, args, max_batches=args.eval_max_batches)
    )
    performance = {
        "method": args.method,
        "protocol": {
            "backbone": "source-pretrained BrainUICL architecture only",
            "continual_learning": METHOD_DESCRIPTIONS[args.method],
            "pseudo_labels": "task-local CPC guide hard argmax",
            "confidence_filter": False,
            "replay": True,
            "replay_ratio": args.replay_ratio,
            "memory_capacity_sequences": args.memory_capacity,
            "memory_admission": (
                "reservoir over every uploaded sequence occurrence"
                if args.method in {"plain_er", "spr_er"}
                else "task-end purity/diversity sequence pruning"
            ),
            "stored_targets": "admission-time hard pseudo labels with optional epoch mask",
            "replay_loss": (
                "hard pseudo-label cross entropy"
                if args.method != "puridiver_cru"
                else "PuriDivER clean/relabel/unlabeled branch loss"
            ),
            "spr_filter": args.method == "spr_er",
            "puridiver_memory": args.method in {"puridiver_memory_ce", "puridiver_cru"},
            "puridiver_cru": args.method == "puridiver_cru",
            "regularization_cl_penalty": False,
            "brainuicl_cea": False,
            "brainuicl_dcb": False,
            "brainuicl_source_replay": False,
            "true_target_labels_used_for_training": False,
            "attack": args.attack_mode,
            "attack_tasks": sorted(args.attack_tasks),
        },
        "config": serializable_args(args),
        "initial": {
            "old_generalization": initial_old,
            "source_train": metric_view(
                evaluate(
                    student_blocks,
                    source_loader,
                    args,
                    max_batches=args.eval_max_batches,
                )
            ),
            "validation": metric_view(
                evaluate(
                    student_blocks,
                    val_loader,
                    args,
                    max_batches=args.eval_max_batches,
                )
            ),
        },
        "stability": {
            "acc": [initial_old["acc"]],
            "mf1": [initial_old["mf1"]],
        },
        "tasks": [],
        "retention_snapshots": {},
        "final": {},
    }
    seen_subjects: list[int] = []
    total_tasks = len(split["new_order"])

    for task_index, subject in enumerate(split["new_order"], start=1):
        fix_randomness(args.seed + 1000 * task_index)
        print(
            f"[{args.method}] task={task_index}/{total_tasks} subject={subject}",
            flush=True,
        )
        clean_data_paths, clean_label_paths = subject_paths(args.data_root, subject)
        clean_eval_loader = make_loader(
            args.data_root,
            [subject],
            args.batch,
            shuffle=False,
            num_workers=args.num_worker,
        )
        before = metric_view(
            evaluate(
                student_blocks,
                clean_eval_loader,
                args,
                max_batches=args.eval_max_batches,
            )
        )
        attack_diagnostics = None
        attack_label_cpc_losses: list[float] = []
        generated_paths: list[Path] = []
        training_data_paths = list(clean_data_paths)
        training_label_paths = list(clean_label_paths)

        if task_index in args.attack_tasks:
            if args.attack_mode == "benign_repeat":
                attack_diagnostics = {
                    "mode": "benign_repeat",
                    "task": task_index,
                    "subject": int(subject),
                    "poisoned_sequences": 0,
                    "total_sequences": len(clean_data_paths),
                    "poison_fraction": 0.0,
                    "poison_indices": list(range(len(clean_data_paths))),
                    "generated_inputs": False,
                    "learner_replay": True,
                    "diagnostics_mean": {
                        "relative_l2_eog": 0.0,
                        "relative_l2_eeg": 0.0,
                        "pseudo_label_preservation": 1.0,
                    },
                }
                poison_indices = attack_diagnostics["poison_indices"]
            elif args.attack_mode == "proxy_dual_harm" and args.fixed_upload_root is not None:
                poisoned_paths, attack_diagnostics = load_fixed_attack_uploads(
                    args,
                    task_index,
                    subject,
                    clean_data_paths,
                )
                training_data_paths = list(poisoned_paths)
                poison_indices = attack_diagnostics["poison_indices"]
                generated_paths = list(poisoned_paths)
                attack_diagnostics["learner_replay"] = True
                attack_diagnostics["regularizer_state_visible_to_attacker"] = False
                attack_diagnostics["replay_memory_visible_to_attacker"] = False
                attack_diagnostics["fixed_upload_manifest_validated"] = True
            elif args.attack_mode == "proxy_dual_harm":
                clean_attack_loader = make_unlabeled_loader(
                    clean_data_paths, args, shuffle=True
                )
                attack_guide, attack_label_cpc_losses = adapt_guiding_model(
                    student_blocks,
                    clean_attack_loader,
                    args,
                    task_index,
                    subject,
                )
                poisoned_paths, attack_diagnostics = (
                    materialize_batched_proxy_dual_harm_subject(
                        student_blocks=student_blocks,
                        label_blocks=attack_guide,
                        strategy=None,
                        current_data_paths=clean_data_paths,
                        reference_data_paths=source_reference_paths,
                        output_dir=(
                            args.output_root
                            / "poisoned_inputs"
                            / f"task_{task_index}_subject_{subject}"
                        ),
                        task_index=task_index,
                        subject=subject,
                        args=args,
                    )
                )
                training_data_paths = list(poisoned_paths)
                poison_indices = attack_diagnostics["poison_indices"]
                generated_paths = [poisoned_paths[index] for index in poison_indices]
                attack_diagnostics["learner_replay"] = True
                attack_diagnostics["regularizer_state_visible_to_attacker"] = False
                attack_diagnostics["replay_memory_visible_to_attacker"] = False
            else:
                raise ValueError(f"Unsupported attack mode: {args.attack_mode}")

            repeat_paths = [training_data_paths[index] for index in poison_indices]
            repeat_labels = [training_label_paths[index] for index in poison_indices]
            training_data_paths.extend(repeat_paths * args.attack_proxy_repeat)
            training_label_paths.extend(repeat_labels * args.attack_proxy_repeat)
            attack_diagnostics.update(
                {
                    "proxy_repeat": args.attack_proxy_repeat,
                    "injected_proxy_sequences": (
                        len(repeat_paths) * args.attack_proxy_repeat
                    ),
                    "training_sequences_after_injection": len(training_data_paths),
                    "memory_admission_candidates": len(training_data_paths),
                }
            )

        fix_randomness(args.seed + 1000 * task_index)
        current_loader = make_unlabeled_loader(
            training_data_paths, args, shuffle=True
        )
        guiding_blocks, cpc_losses = adapt_guiding_model(
            student_blocks,
            current_loader,
            args,
            task_index,
            subject,
        )
        training = train_er_task(
            student_blocks,
            guiding_blocks,
            current_loader,
            memory,
            args,
            task_index,
            subject,
        )
        admission_labels = infer_pseudo_labels(
            guiding_blocks, training_data_paths, args
        )
        poisoned_set = {str(path) for path in generated_paths}
        records = build_memory_records(
            training_data_paths,
            admission_labels,
            task_index=task_index,
            subject=subject,
            original_count=len(clean_data_paths),
            poisoned_paths=poisoned_set,
        )
        memory_update, admission_diagnostics, retained_records = admit_memory(
            memory,
            records,
            training_data_paths,
            student_blocks,
            args,
        )

        diagnostic_loader = DataLoader(
            # Labels are opened only for the following offline diagnostic.
            SequenceDataset((training_data_paths, training_label_paths)),
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_worker,
        )
        pseudo_diagnostics = pseudo_label_diagnostics(
            guiding_blocks, diagnostic_loader, args
        )
        clean_pseudo_diagnostics = pseudo_label_diagnostics(
            guiding_blocks, clean_eval_loader, args
        )
        after = metric_view(
            evaluate(
                student_blocks,
                clean_eval_loader,
                args,
                max_batches=args.eval_max_batches,
            )
        )
        old = metric_view(
            evaluate(
                student_blocks,
                old_loader,
                args,
                max_batches=args.eval_max_batches,
            )
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
            "training": training,
            "memory_update": memory_update,
            "admission_diagnostics": admission_diagnostics,
            "memory": memory.stats(),
            "attack": attack_diagnostics,
            "attack_label_guiding_cpc_losses": attack_label_cpc_losses,
        }
        performance["tasks"].append(task_row)
        if task_index in args.retention_milestones or task_index == total_tasks:
            performance["retention_snapshots"][str(task_index)] = (
                evaluate_seen_subjects(student_blocks, seen_subjects, args)
            )
        save_json(args.output_root / "metrics.json", performance)
        print(
            f"[{args.method}] subject={subject} current "
            f"ACC {before['acc']:.4f}->{after['acc']:.4f}; "
            f"old ACC={old['acc']:.4f}; memory={len(memory)} "
            f"poison_memory={memory.stats()['poisoned_fraction']:.4f}",
            flush=True,
        )

    final_seen = performance["retention_snapshots"][str(total_tasks)]
    performance["final"] = {
        "old_generalization": metric_view(
            evaluate(
                student_blocks,
                old_loader,
                args,
                max_batches=args.eval_max_batches,
            )
        ),
        "source_train": metric_view(
            evaluate(
                student_blocks,
                source_loader,
                args,
                max_batches=args.eval_max_batches,
            )
        ),
        "validation": metric_view(
            evaluate(
                student_blocks,
                val_loader,
                args,
                max_batches=args.eval_max_batches,
            )
        ),
        "seen_subjects": final_seen,
        "initial_model_seen_subjects": evaluate_seen_subjects(
            initial_blocks, split["new_order"], args
        ),
        "memory": memory.stats(),
        "memory_records": memory.serializable_records(),
    }
    summary = summarize_run(performance)
    summary.update(
        {
            "final_memory_size": len(memory),
            "final_memory_poisoned_fraction": memory.stats()[
                "poisoned_fraction"
            ],
            "replay_poisoned_fraction": memory.stats()[
                "poisoned_replay_fraction"
            ],
            "total_replay_draws": memory.stats()["total_replay_draws"],
        }
    )
    performance["summary"] = summary
    save_json(args.output_root / "metrics.json", performance)
    save_json(args.output_root / "summary.json", {args.method: summary})
    return performance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=REPLAY_METHODS, default="plain_er")
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
        default=REPO_ROOT / "experiments" / "replay_cl_eeg_runs" / "latest",
    )
    parser.add_argument(
        "--fixed-upload-root",
        type=Path,
        default=None,
        help="attack_shared run root containing poisoned_inputs and metrics.json",
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
    parser.add_argument("--freeze-bn-stats", action="store_true")
    parser.add_argument("--memory-capacity", type=int, default=1000)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--attack-mode", choices=ATTACK_MODES, default="none")
    parser.add_argument("--attack-tasks", type=str, default="")
    parser.add_argument("--attack-fraction", type=float, default=1.0)
    parser.add_argument("--attack-eps-scale", type=float, default=0.5)
    parser.add_argument("--attack-max-relative-l2", type=float, default=0.2)
    parser.add_argument("--attack-steps", type=int, default=3)
    parser.add_argument("--attack-inner-lr", type=float, default=1e-4)
    parser.add_argument("--attack-param-scope", default="classifier")
    parser.add_argument("--attack-reference-batch", type=int, default=4)
    parser.add_argument("--attack-generation-batch", type=int, default=4)
    parser.add_argument("--attack-random-start", action="store_true")
    parser.add_argument("--attack-target-weight", type=float, default=5.0)
    parser.add_argument("--attack-conflict-weight", type=float, default=1.0)
    parser.add_argument("--attack-gradient-norm-weight", type=float, default=0.25)
    parser.add_argument("--attack-virtual-old-weight", type=float, default=1.0)
    parser.add_argument("--attack-virtual-new-weight", type=float, default=1.0)
    parser.add_argument("--attack-new-proxy-weight", type=float, default=1.0)
    parser.add_argument("--attack-curvature-scale", type=float, default=0.0)
    parser.add_argument("--attack-min-confidence", type=float, default=0.85)
    parser.add_argument("--attack-confidence-weight", type=float, default=2.0)
    parser.add_argument("--attack-l2-weight", type=float, default=0.01)
    parser.add_argument("--attack-proxy-repeat", type=int, default=3)
    parser.add_argument("--spr-ensembles", type=int, default=5)
    parser.add_argument("--spr-bmm-iters", type=int, default=10)
    parser.add_argument("--puridiver-clean-threshold", type=float, default=0.5)
    parser.add_argument("--puridiver-uncertainty-threshold", type=float, default=0.5)
    parser.add_argument("--puridiver-diversity-coefficient", type=float, default=0.4)
    parser.add_argument("--puridiver-strong-noise", type=float, default=0.01)
    parser.add_argument("--puridiver-strong-scale", type=float, default=0.08)
    parser.add_argument("--puridiver-strong-mask-fraction", type=float, default=0.0)
    parser.add_argument("--puridiver-consistency-weight", type=float, default=1.0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--retention-milestones", type=str, default="10,25,49")
    args = parser.parse_args()

    if args.fixed_upload_root is not None:
        args.fixed_upload_root = args.fixed_upload_root.resolve()
    if args.memory_capacity < 1:
        parser.error("--memory-capacity must be positive")
    if args.replay_ratio != 1.0:
        parser.error("This aligned baseline currently requires --replay-ratio 1.0")
    if not 0.0 <= args.attack_fraction <= 1.0:
        parser.error("--attack-fraction must be in [0, 1]")
    if args.attack_proxy_repeat < 0:
        parser.error("--attack-proxy-repeat must be non-negative")
    if args.spr_ensembles < 1 or args.spr_bmm_iters < 1:
        parser.error("SPR ensemble and BMM iterations must be positive")
    if not 0.0 <= args.puridiver_clean_threshold <= 1.0:
        parser.error("--puridiver-clean-threshold must be in [0, 1]")
    if not 0.0 <= args.puridiver_uncertainty_threshold <= 1.0:
        parser.error("--puridiver-uncertainty-threshold must be in [0, 1]")
    if not 0.0 <= args.puridiver_diversity_coefficient <= 1.0:
        parser.error("--puridiver-diversity-coefficient must be in [0, 1]")
    if args.puridiver_strong_noise < 0 or args.puridiver_strong_scale < 0:
        parser.error("PuriDivER augmentation strengths must be non-negative")
    if not 0.0 <= args.puridiver_strong_mask_fraction <= 1.0:
        parser.error("--puridiver-strong-mask-fraction must be in [0, 1]")
    if args.attack_mode == "none" and args.attack_tasks.strip():
        parser.error("--attack-tasks requires a non-none attack mode")
    if args.attack_mode != "none" and not args.attack_tasks.strip():
        parser.error("A non-none attack mode requires --attack-tasks")

    args.dataset = "ISRUC"
    args.model_param = ModelConfig(args.dataset)
    args.device = torch.device(
        f"cuda:{args.gpu}"
        if args.gpu >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    args.retention_milestones = parse_int_set(args.retention_milestones)
    return args


if __name__ == "__main__":
    run(parse_args())
