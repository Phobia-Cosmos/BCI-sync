#!/usr/bin/env python3
"""Plain experience replay on the aligned unlabeled ISRUC CL protocol.

This runner deliberately reuses the source-pretrained BrainUICL network as a
backbone only.  The continual-learning algorithm is plain reservoir ER:

* no BrainUICL confidence gate, CEA, DCB, source replay, or growing buffer;
* no EWC/SI/MAS parameter penalty;
* current targets are hard labels from the task-local CPC guide;
* previous sequences are sampled 1:1 from a fixed reservoir with their stored
  admission-time pseudo labels.

The clean, benign-repeat, and proxy-dual-harm modes match the regularization
runner's upload protocol so replay persistence can be tested directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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


@dataclass
class ReplayRecord:
    data_path: Path
    pseudo_labels: np.ndarray
    task: int
    subject: int
    sequence_index: int
    poisoned: bool
    repeated_upload: bool
    replay_count: int = 0

    def serializable(self) -> dict:
        return {
            "data_path": str(self.data_path),
            "pseudo_labels": self.pseudo_labels.astype(int).tolist(),
            "task": self.task,
            "subject": self.subject,
            "sequence_index": self.sequence_index,
            "poisoned": self.poisoned,
            "repeated_upload": self.repeated_upload,
            "replay_count": self.replay_count,
        }


class ReservoirReplayMemory:
    """Fixed-capacity sequence reservoir with independent deterministic RNG."""

    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.records: list[ReplayRecord] = []
        self.total_seen = 0
        self.rng = np.random.default_rng(seed)
        self.total_replay_draws = 0
        self.poisoned_replay_draws = 0

    def __len__(self) -> int:
        return len(self.records)

    def add(self, incoming: Sequence[ReplayRecord]) -> dict:
        inserted = 0
        replaced = 0
        discarded = 0
        for record in incoming:
            self.total_seen += 1
            if len(self.records) < self.capacity:
                self.records.append(record)
                inserted += 1
                continue
            location = int(self.rng.integers(0, self.total_seen))
            if location < self.capacity:
                self.records[location] = record
                replaced += 1
            else:
                discarded += 1
        return {
            "candidates": len(incoming),
            "inserted": inserted,
            "replaced": replaced,
            "discarded": discarded,
            "total_seen": self.total_seen,
            "size": len(self.records),
        }

    def sample(self, count: int) -> list[ReplayRecord]:
        if not self.records or count <= 0:
            return []
        replace = len(self.records) < count
        indices = self.rng.choice(len(self.records), count, replace=replace)
        selected = [self.records[int(index)] for index in indices]
        for record in selected:
            record.replay_count += 1
        poisoned = sum(int(record.poisoned) for record in selected)
        self.total_replay_draws += len(selected)
        self.poisoned_replay_draws += poisoned
        return selected

    def stats(self) -> dict:
        poisoned = sum(int(record.poisoned) for record in self.records)
        repeated = sum(int(record.repeated_upload) for record in self.records)
        return {
            "capacity": self.capacity,
            "size": len(self.records),
            "total_seen": self.total_seen,
            "unique_paths": len({str(record.data_path) for record in self.records}),
            "poisoned_records": poisoned,
            "poisoned_fraction": poisoned / max(len(self.records), 1),
            "repeated_upload_records": repeated,
            "repeated_upload_fraction": repeated / max(len(self.records), 1),
            "total_replay_draws": self.total_replay_draws,
            "poisoned_replay_draws": self.poisoned_replay_draws,
            "poisoned_replay_fraction": (
                self.poisoned_replay_draws / max(self.total_replay_draws, 1)
            ),
        }

    def serializable_records(self) -> list[dict]:
        return [record.serializable() for record in self.records]


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


def load_replay_batch(
    records: Sequence[ReplayRecord],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = [
        torch.from_numpy(np.load(record.data_path).astype(np.float32))
        for record in records
    ]
    values = torch.stack(arrays)
    labels = torch.stack(
        [torch.from_numpy(record.pseudo_labels.astype(np.int64)) for record in records]
    )
    return values[:, :, :2, :], values[:, :, 2:, :], labels


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
    memory: ReservoirReplayMemory,
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

        for eog, eeg, _dummy in current_loader:
            eog = eog.to(args.device)
            eeg = eeg.to(args.device)
            with torch.no_grad():
                current_labels = flat_logits(
                    forward_blocks(guiding_blocks, eog, eeg, args)
                ).argmax(dim=1)
            current_logits = flat_logits(
                forward_blocks(student_blocks, eog, eeg, args)
            )
            current_loss = F.cross_entropy(current_logits, current_labels)

            replay_records = memory.sample(eog.shape[0])
            if replay_records:
                replay_eog, replay_eeg, replay_labels = load_replay_batch(
                    replay_records
                )
                replay_eog = replay_eog.to(args.device)
                replay_eeg = replay_eeg.to(args.device)
                replay_labels = replay_labels.reshape(-1).to(args.device)
                replay_logits = flat_logits(
                    forward_blocks(student_blocks, replay_eog, replay_eeg, args)
                )
                replay_loss = F.cross_entropy(replay_logits, replay_labels)
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


def build_memory_records(
    data_paths: Sequence[Path],
    pseudo_labels: Sequence[np.ndarray],
    *,
    task_index: int,
    subject: int,
    original_count: int,
    poisoned_paths: set[str],
) -> list[ReplayRecord]:
    if len(data_paths) != len(pseudo_labels):
        raise ValueError("Replay admission data/label count mismatch")
    records: list[ReplayRecord] = []
    for upload_index, (path, labels) in enumerate(zip(data_paths, pseudo_labels)):
        sequence_index = int(path.stem)
        records.append(
            ReplayRecord(
                data_path=Path(path),
                pseudo_labels=np.asarray(labels, dtype=np.int64),
                task=int(task_index),
                subject=int(subject),
                sequence_index=sequence_index,
                poisoned=str(Path(path)) in poisoned_paths,
                repeated_upload=upload_index >= original_count,
            )
        )
    return records


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
    memory = ReservoirReplayMemory(args.memory_capacity, args.seed + 700_001)
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
        "method": "plain_er",
        "protocol": {
            "backbone": "source-pretrained BrainUICL architecture only",
            "continual_learning": "fixed-capacity reservoir experience replay",
            "pseudo_labels": "task-local CPC guide hard argmax",
            "confidence_filter": False,
            "replay": True,
            "replay_ratio": args.replay_ratio,
            "memory_capacity_sequences": args.memory_capacity,
            "memory_admission": "reservoir over every uploaded sequence occurrence",
            "stored_targets": "admission-time hard pseudo labels",
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
            f"[plain_er] task={task_index}/{total_tasks} subject={subject}",
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
        memory_update = memory.add(records)

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
            f"[plain_er] subject={subject} current "
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
    save_json(args.output_root / "summary.json", {"plain_er": summary})
    return performance


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
        default=REPO_ROOT / "experiments" / "replay_cl_eeg_runs" / "latest",
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
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--retention-milestones", type=str, default="10,25,49")
    args = parser.parse_args()

    if args.memory_capacity < 1:
        parser.error("--memory-capacity must be positive")
    if args.replay_ratio != 1.0:
        parser.error("This aligned baseline currently requires --replay-ratio 1.0")
    if not 0.0 <= args.attack_fraction <= 1.0:
        parser.error("--attack-fraction must be in [0, 1]")
    if args.attack_proxy_repeat < 0:
        parser.error("--attack-proxy-repeat must be non-negative")
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
