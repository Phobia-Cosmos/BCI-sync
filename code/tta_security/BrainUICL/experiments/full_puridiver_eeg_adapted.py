#!/usr/bin/env python3
"""Full PuriDivER lifecycle adapted to unlabeled ISRUC subject streams."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.full_puridiver_eeg import (  # noqa: E402
    CRUPartition,
    DynamicPuriMemory,
    PuriMemoryScores,
    PuriSequenceRecord,
    build_cru_partition,
)
from model.full_spr_eeg import augment_eeg_views  # noqa: E402
from model.regularization_cl import freeze_batch_norm_running_stats  # noqa: E402
from n2n_shared_proxy import (  # noqa: E402
    resolve_task as resolve_n2n_task,
    sha256_file,
    validate_manifest,
)
from regularization_cl_eeg import (  # noqa: E402
    adapt_guiding_model,
    build_split,
    evaluate_seen_subjects,
    metric_view,
    parse_int_set,
    pseudo_label_diagnostics,
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
from unlabeled_eeg import UnlabeledSequenceDataset  # noqa: E402
from utils.config import ModelConfig  # noqa: E402
from utils.util import fix_randomness  # noqa: E402


STATE_SCHEMA = "full-puridiver-eeg-adapted-state-v1"


class CurrentPseudoDataset(Dataset):
    def __init__(self, paths: Sequence[Path], pseudo_labels: Sequence[np.ndarray]):
        if len(paths) != len(pseudo_labels):
            raise ValueError("Current PuriDivER signals and pseudo labels differ")
        self.paths = [Path(path) for path in paths]
        self.pseudo_labels = [
            np.asarray(labels, dtype=np.int64) for labels in pseudo_labels
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        values = torch.from_numpy(
            np.load(self.paths[index], allow_pickle=False).astype(
                np.float32, copy=False
            )
        )
        return (
            values[:, :2, :],
            values[:, 2:, :],
            torch.from_numpy(self.pseudo_labels[index]),
            index,
        )


class PuriRecordDataset(Dataset):
    def __init__(self, records: Sequence[PuriSequenceRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        values = torch.from_numpy(
            np.load(record.data_path, allow_pickle=False).astype(
                np.float32, copy=False
            )
        )
        return (
            values[:, :2, :],
            values[:, 2:, :],
            torch.from_numpy(record.pseudo_labels.copy()),
            torch.from_numpy(record.epoch_mask.copy()),
            index,
        )


class WeightedReplayDataset(Dataset):
    def __init__(self, groups: Sequence[tuple[PuriSequenceRecord, np.ndarray]]):
        self.groups = list(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int):
        record, weights = self.groups[index]
        values = torch.from_numpy(
            np.load(record.data_path, allow_pickle=False).astype(
                np.float32, copy=False
            )
        )
        return (
            values[:, :2, :],
            values[:, 2:, :],
            torch.from_numpy(record.pseudo_labels.copy()),
            torch.from_numpy(np.asarray(weights, dtype=np.float32)),
        )


class PartitionedMemoryDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PuriSequenceRecord],
        categories: Sequence[np.ndarray],
        clean_probabilities: Sequence[np.ndarray],
        snapshot_probabilities: Sequence[np.ndarray],
    ):
        self.records = list(records)
        self.categories = list(categories)
        self.clean_probabilities = list(clean_probabilities)
        self.snapshot_probabilities = list(snapshot_probabilities)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        values = torch.from_numpy(
            np.load(record.data_path, allow_pickle=False).astype(
                np.float32, copy=False
            )
        )
        return (
            values[:, :2, :],
            values[:, 2:, :],
            torch.from_numpy(record.pseudo_labels.copy()),
            torch.from_numpy(self.categories[index].copy()),
            torch.from_numpy(self.clean_probabilities[index].copy()),
            torch.from_numpy(self.snapshot_probabilities[index].copy()),
        )


def serializable_args(args) -> dict[str, Any]:
    payload = vars(args).copy()
    payload["device"] = str(args.device)
    payload["model_param"] = "ModelConfig(ISRUC)"
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, set):
            payload[key] = sorted(value)
    return payload


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_unlabeled_loader(
    paths: Sequence[Path], args, *, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        UnlabeledSequenceDataset(paths, args.model_param.SeqLength),
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=args.num_worker,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def infer_pseudo_labels(blocks, paths: Sequence[Path], args) -> list[np.ndarray]:
    loader = make_unlabeled_loader(paths, args, shuffle=False, seed=args.seed)
    set_train(blocks, False)
    rows: list[np.ndarray] = []
    for eog, eeg, _dummy in loader:
        logits = forward_blocks(
            blocks, eog.to(args.device), eeg.to(args.device), args
        )
        rows.extend(logits.argmax(dim=1).detach().cpu().numpy())
    return [np.asarray(row, dtype=np.int64) for row in rows]


def penultimate_forward(blocks, eog, eeg, args):
    batch = eeg.shape[0]
    eog = eog.reshape(
        -1, args.model_param.EogNum, args.model_param.EpochLength
    )
    eeg = eeg.reshape(
        -1, args.model_param.EegNum, args.model_param.EpochLength
    )
    features = blocks[0](eeg, eog)
    features = blocks[1](features)
    penultimate = blocks[2].sleep_stage_mlp(features)
    logits = blocks[2].sleep_stage_classifier(penultimate).permute(0, 2, 1)
    return (
        logits.reshape(batch, args.model_param.NumClasses, args.model_param.SeqLength),
        penultimate.reshape(batch, args.model_param.SeqLength, -1),
    )


def weighted_ce_sum(logits, labels, weights):
    per_epoch = F.cross_entropy(
        flat_logits(logits), labels.reshape(-1).long(), reduction="none"
    ).reshape_as(weights)
    return (per_epoch * weights).sum(), weights.sum()


def make_optimizer(blocks, args):
    return torch.optim.Adam(
        [parameter for block in blocks for parameter in block.parameters()],
        lr=args.cl_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )


@torch.no_grad()
def infer_memory_snapshot(blocks, records, args) -> dict[str, np.ndarray]:
    loader = DataLoader(
        PuriRecordDataset(records),
        batch_size=args.infer_batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    set_train(blocks, False)
    losses: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    features: list[np.ndarray] = []
    for eog, eeg, labels, masks, _indices in loader:
        logits, penultimate = penultimate_forward(
            blocks, eog.to(args.device), eeg.to(args.device), args
        )
        labels = labels.to(args.device)
        masks = masks.bool().to(args.device)
        flat_loss = F.cross_entropy(
            flat_logits(logits), labels.reshape(-1), reduction="none"
        ).reshape(labels.shape)
        probability = logits.softmax(dim=1).permute(0, 2, 1)
        losses.append(flat_loss[masks].cpu().numpy())
        probabilities.append(probability[masks].cpu().numpy())
        features.append(penultimate[masks].cpu().numpy())
    return {
        "losses": np.concatenate(losses),
        "probabilities": np.concatenate(probabilities),
        "features": np.concatenate(features),
        "classifier_weights": blocks[2]
        .sleep_stage_classifier.weight.detach()
        .cpu()
        .numpy(),
    }


def score_memory(blocks, records, args) -> PuriMemoryScores:
    snapshot = infer_memory_snapshot(blocks, records, args)
    return PuriMemoryScores(
        losses=snapshot["losses"],
        features=snapshot["features"],
        classifier_weights=snapshot["classifier_weights"],
    )


def _backward_replay_groups(
    blocks,
    groups: Sequence[tuple[PuriSequenceRecord, np.ndarray]],
    denominator: int,
    scale: float,
    args,
) -> float:
    if not groups:
        return 0.0
    loader = DataLoader(
        WeightedReplayDataset(groups),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    detached_sum = 0.0
    for eog, eeg, labels, weights in loader:
        logits = forward_blocks(
            blocks, eog.to(args.device), eeg.to(args.device), args
        )
        loss_sum, _count = weighted_ce_sum(
            logits, labels.to(args.device), weights.to(args.device)
        )
        (scale * loss_sum / max(denominator, 1)).backward()
        detached_sum += float(loss_sum.detach().cpu())
    return detached_sum


def online_train_subject(
    blocks,
    optimizer,
    paths: Sequence[Path],
    pseudo_labels: Sequence[np.ndarray],
    clean_data_paths: Sequence[Path],
    proxy_indices: set[int],
    memory: DynamicPuriMemory,
    args,
    task_index: int,
    subject: int,
) -> dict[str, Any]:
    loader = DataLoader(
        CurrentPseudoDataset(paths, pseudo_labels),
        batch_size=args.online_batch_sequences,
        shuffle=False,
        num_workers=args.num_worker,
    )
    rows: list[dict[str, Any]] = []
    for batch_index, (eog, eeg, labels, slots) in enumerate(loader):
        if args.max_online_batches and batch_index >= args.max_online_batches:
            break
        set_train(blocks, True)
        if args.freeze_student_bn_stats:
            freeze_batch_norm_running_stats(blocks)
        eog = eog.to(args.device)
        eeg = eeg.to(args.device)
        labels_device = labels.to(args.device)
        logits = forward_blocks(blocks, eog, eeg, args)
        current_weights = torch.ones_like(labels_device, dtype=torch.float32)
        current_sum, current_count = weighted_ce_sum(
            logits, labels_device, current_weights
        )
        current_mean = current_sum / current_count.clamp_min(1)
        replay_groups = (
            memory.sample_epoch_weights(int(current_count.item()))
            if task_index > 1 and len(memory)
            else []
        )
        replay_count = sum(
            int(weights.sum()) for _record, weights in replay_groups
        )
        optimizer.zero_grad(set_to_none=True)
        if replay_count:
            (0.5 * current_mean).backward()
            replay_sum = _backward_replay_groups(
                blocks, replay_groups, replay_count, 0.5, args
            )
            online_loss_value = 0.5 * float(current_mean.detach().cpu()) + 0.5 * (
                replay_sum / replay_count
            )
        else:
            current_mean.backward()
            replay_sum = 0.0
            online_loss_value = float(current_mean.detach().cpu())
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for block in blocks for parameter in block.parameters()],
                args.grad_clip,
            )
        optimizer.step()

        incoming: list[PuriSequenceRecord] = []
        for row, slot_tensor in enumerate(slots):
            slot = int(slot_tensor)
            incoming.append(
                PuriSequenceRecord(
                    data_path=paths[slot],
                    pseudo_labels=np.asarray(pseudo_labels[slot], dtype=np.int64),
                    epoch_mask=np.ones(
                        args.model_param.SeqLength, dtype=bool
                    ),
                    task=task_index,
                    subject=subject,
                    sequence_index=int(clean_data_paths[slot].stem),
                    is_proxy=slot in proxy_indices,
                )
            )
        coefficient = args.max_diversity_coefficient * min(
            1.0 / max(online_loss_value, 1e-8), 1.0
        )
        update = memory.update(
            incoming,
            lambda records: score_memory(blocks, records, args),
            coefficient,
        )
        rows.append(
            {
                "mini_batch": batch_index + 1,
                "current_sequences": len(slots),
                "current_epochs": int(current_count.item()),
                "replay_epochs": replay_count,
                "current_ce": float(current_mean.detach().cpu()),
                "replay_ce": replay_sum / max(replay_count, 1),
                "online_loss": online_loss_value,
                "diversity_coefficient": coefficient,
                "memory_update": update,
            }
        )
    return {
        "mini_batches": len(rows),
        "rows": rows,
        "memory_updates": len(rows),
        "mean_online_loss": float(np.mean([row["online_loss"] for row in rows])),
        "mean_diversity_coefficient": float(
            np.mean([row["diversity_coefficient"] for row in rows])
        ),
    }


def train_memory_ce(blocks, optimizer, memory, args, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PuriRecordDataset(memory.records),
        batch_size=args.replay_batch_sequences,
        shuffle=True,
        num_workers=args.num_worker,
        generator=generator,
    )
    losses: list[float] = []
    for batch_index, (eog, eeg, labels, masks, _indices) in enumerate(loader):
        if args.max_replay_batches and batch_index >= args.max_replay_batches:
            break
        set_train(blocks, True)
        if args.freeze_student_bn_stats:
            freeze_batch_norm_running_stats(blocks)
        logits = forward_blocks(
            blocks, eog.to(args.device), eeg.to(args.device), args
        )
        loss_sum, count = weighted_ce_sum(
            logits,
            labels.to(args.device),
            masks.float().to(args.device),
        )
        loss = loss_sum / count.clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for block in blocks for parameter in block.parameters()],
                args.grad_clip,
            )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def partition_by_record(
    memory: DynamicPuriMemory,
    partition: CRUPartition,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    categories: list[np.ndarray] = []
    clean_probabilities: list[np.ndarray] = []
    snapshot_probabilities: list[np.ndarray] = []
    offset = 0
    for record in memory.records:
        active_indices = np.flatnonzero(record.epoch_mask)
        count = active_indices.size
        category = np.full(record.pseudo_labels.size, -1, dtype=np.int64)
        clean = np.zeros(record.pseudo_labels.size, dtype=np.float32)
        snapshot = np.zeros(
            (record.pseudo_labels.size, partition.snapshot_probability.shape[1]),
            dtype=np.float32,
        )
        local = slice(offset, offset + count)
        local_clean = partition.clean_mask[local]
        local_relabel = partition.relabel_mask[local]
        local_unlabeled = partition.unlabeled_mask[local]
        category[active_indices[local_clean]] = 0
        category[active_indices[local_relabel]] = 1
        category[active_indices[local_unlabeled]] = 2
        clean[active_indices] = partition.clean_probability[local]
        snapshot[active_indices] = partition.snapshot_probability[local]
        categories.append(category)
        clean_probabilities.append(clean)
        snapshot_probabilities.append(snapshot)
        offset += count
    if offset != partition.clean_mask.size:
        raise RuntimeError("PuriDivER partition does not cover the physical memory")
    return categories, clean_probabilities, snapshot_probabilities


def train_cru_epoch(blocks, optimizer, memory, partition, args, seed: int) -> float:
    categories, clean_probabilities, snapshot_probabilities = partition_by_record(
        memory, partition
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PartitionedMemoryDataset(
            memory.records,
            categories,
            clean_probabilities,
            snapshot_probabilities,
        ),
        batch_size=args.replay_batch_sequences,
        shuffle=True,
        num_workers=args.num_worker,
        generator=generator,
    )
    losses: list[float] = []
    for batch_index, (eog, eeg, labels, category, clean_p, snapshot_p) in enumerate(loader):
        if args.max_replay_batches and batch_index >= args.max_replay_batches:
            break
        set_train(blocks, True)
        if args.freeze_student_bn_stats:
            freeze_batch_norm_running_stats(blocks)
        device = args.device
        eog, eeg = eog.to(device), eeg.to(device)
        labels = labels.to(device)
        category = category.to(device)
        clean_p = clean_p.to(device)
        snapshot_p = snapshot_p.to(device)
        logits = forward_blocks(blocks, eog, eeg, args)
        flat = flat_logits(logits)
        flat_labels = labels.reshape(-1)
        flat_category = category.reshape(-1)
        numerator = torch.zeros((), device=device)
        active_count = (flat_category >= 0).sum()
        clean_mask = flat_category == 0
        relabel_mask = flat_category == 1
        unlabeled_mask = flat_category == 2
        if clean_mask.any():
            numerator = numerator + F.cross_entropy(
                flat[clean_mask], flat_labels[clean_mask], reduction="sum"
            )
        if relabel_mask.any():
            observed = F.one_hot(
                flat_labels[relabel_mask], args.model_param.NumClasses
            ).float()
            w = clean_p.reshape(-1)[relabel_mask].unsqueeze(1)
            target = w * observed + (1.0 - w) * snapshot_p.reshape(
                -1, args.model_param.NumClasses
            )[relabel_mask]
            numerator = numerator - (
                target * F.log_softmax(flat[relabel_mask], dim=1)
            ).sum()
        if unlabeled_mask.any():
            weak_eog, weak_eeg = augment_eeg_views(
                eog,
                eeg,
                jitter=args.weak_noise,
                scale=args.weak_scale,
                mask_ratio=0.0,
                channel_drop=0.0,
            )
            strong_eog, strong_eeg = augment_eeg_views(
                eog,
                eeg,
                jitter=args.strong_noise,
                scale=args.strong_scale,
                mask_ratio=args.strong_mask_fraction,
                channel_drop=args.strong_channel_drop,
            )
            with torch.no_grad():
                weak_probability = forward_blocks(
                    blocks, weak_eog, weak_eeg, args
                ).softmax(dim=1)
            strong_probability = forward_blocks(
                blocks, strong_eog, strong_eeg, args
            ).softmax(dim=1)
            weak_flat = weak_probability.permute(0, 2, 1).reshape(
                -1, args.model_param.NumClasses
            )
            strong_flat = strong_probability.permute(0, 2, 1).reshape(
                -1, args.model_param.NumClasses
            )
            consistency = (strong_flat - weak_flat).pow(2).mean(dim=1)
            numerator = numerator + args.consistency_weight * consistency[
                unlabeled_mask
            ].sum()
        loss = numerator / active_count.clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for block in blocks for parameter in block.parameters()],
                args.grad_clip,
            )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def replay_train(blocks, optimizer, memory, args, task_index: int) -> list[dict]:
    rows: list[dict[str, Any]] = []
    for epoch in range(args.replay_epochs):
        seed = args.seed + 100_000 * task_index + epoch
        if epoch < args.warmup_epochs:
            loss = train_memory_ce(blocks, optimizer, memory, args, seed)
            rows.append(
                {"epoch": epoch + 1, "mode": "warmup_hard_ce", "loss": loss}
            )
            continue
        snapshot = infer_memory_snapshot(blocks, memory.records, args)
        partition = build_cru_partition(
            snapshot["losses"],
            snapshot["probabilities"],
            seed=seed,
            min_gmm_samples=args.min_gmm_samples,
        )
        loss = train_cru_epoch(
            blocks, optimizer, memory, partition, args, seed
        )
        rows.append(
            {
                "epoch": epoch + 1,
                "mode": "recomputed_cru",
                "loss": loss,
                **partition.diagnostics,
            }
        )
    return rows


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_state(
    path: Path,
    *,
    split: dict,
    completed_tasks: int,
    blocks,
    optimizer,
    memory: DynamicPuriMemory,
    performance: dict,
    manifest_sha256: str | None,
) -> None:
    payload = {
        "schema": STATE_SCHEMA,
        "split": split,
        "completed_tasks": int(completed_tasks),
        "student_blocks": [block.state_dict() for block in blocks],
        "optimizer": optimizer.state_dict(),
        "memory": memory.state_dict(),
        "rng": _capture_rng_state(),
        "performance": performance,
        "manifest_sha256": manifest_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_state(path, *, split, blocks, optimizer, args, manifest_sha256):
    payload = torch.load(path, map_location=args.device, weights_only=False)
    if payload.get("schema") != STATE_SCHEMA:
        raise ValueError(f"Unsupported Full PuriDivER state: {payload.get('schema')}")
    if payload["split"]["new_order"] != split["new_order"]:
        raise ValueError("Full PuriDivER resume split/order differs")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Full PuriDivER resume manifest differs")
    for block, state in zip(blocks, payload["student_blocks"]):
        block.load_state_dict(state)
    optimizer.load_state_dict(payload["optimizer"])
    memory = DynamicPuriMemory.from_state_dict(payload["memory"])
    _restore_rng_state(payload["rng"])
    return int(payload["completed_tasks"]), memory, payload["performance"]


def resolve_uploaded_paths(args, task_index, subject, clean_data_paths):
    if args.n2n_manifest is None:
        return list(clean_data_paths), (), None
    resolved = resolve_n2n_task(
        args.n2n_manifest,
        task=task_index,
        subject=subject,
        clean_data_paths=clean_data_paths,
        verify=args.n2n_verify,
    )
    return list(resolved.data_paths), resolved.proxy_indices, dict(resolved.diagnostics)


def run(args) -> dict[str, Any]:
    fix_randomness(args.seed)
    split = build_split(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(args.output_root / "split.json", split)
    save_json(args.output_root / "config.json", serializable_args(args))
    manifest_sha256 = (
        sha256_file(args.n2n_manifest) if args.n2n_manifest is not None else None
    )
    manifest_validation = None
    if args.n2n_manifest is not None and not args.skip_n2n_prevalidation:
        expected = {
            task: (int(subject), subject_paths(args.data_root, int(subject))[0])
            for task, subject in enumerate(split["new_order"], start=1)
        }
        manifest_validation = validate_manifest(args.n2n_manifest, expected)
        save_json(args.output_root / "manifest_validation.json", manifest_validation)

    initial_blocks = load_pretrained(args)
    student_blocks = tuple(copy.deepcopy(block).to(args.device) for block in initial_blocks)
    optimizer = make_optimizer(student_blocks, args)
    memory = DynamicPuriMemory(
        args.memory_capacity_epochs, args.model_param.NumClasses, args.seed + 31
    )
    completed_tasks = 0
    old_loader = make_loader(
        args.data_root,
        split["old_idx"],
        args.eval_batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    source_loader = make_loader(
        args.data_root,
        split["train_idx"],
        args.eval_batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    val_loader = make_loader(
        args.data_root,
        split["val_idx"],
        args.eval_batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    initial_old = metric_view(
        evaluate(
            student_blocks,
            old_loader,
            args,
            max_batches=args.eval_max_batches,
        )
    )
    performance = {
        "method": "full_puridiver_eeg_adapted",
        "protocol": {
            "backbone": "BrainUICL FeatureExtractor + TransformerEncoder + SleepMLP",
            "initialization": "common source-supervised checkpoint",
            "student_persistent": True,
            "target_training_loader": "signal-only without annotation paths",
            "pseudo_labels": "task-local CPC guide hard argmax for every epoch",
            "confidence_filter": False,
            "online_update": "current plus 1:1 epoch-weighted memory replay hard CE",
            "memory_update_cadence": "immediately after every online mini-batch",
            "memory_capacity_epochs": args.memory_capacity_epochs,
            "memory_storage": "full sequence plus retained epoch mask",
            "memory_pruning": "majority pseudo-class purity-diversity, recomputed after each deletion",
            "robust_replay": "warmup then per-epoch loss-GMM and uncertainty-GMM C/R/U",
            "cru_masks_persistent": False,
            "source_memory": False,
            "true_target_labels_used_for_training": False,
            "canonical_n2n_manifest": (
                None if args.n2n_manifest is None else str(args.n2n_manifest)
            ),
            "manifest_validation": manifest_validation,
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
        "stability": {"acc": [initial_old["acc"]], "mf1": [initial_old["mf1"]]},
        "tasks": [],
        "retention_snapshots": {},
        "final": {},
    }
    if args.resume_state is not None:
        completed_tasks, memory, performance = load_state(
            args.resume_state,
            split=split,
            blocks=student_blocks,
            optimizer=optimizer,
            args=args,
            manifest_sha256=manifest_sha256,
        )
        performance.pop("interrupted_after_task", None)

    seen_subjects = [
        int(subject) for subject in split["new_order"][:completed_tasks]
    ]
    total_tasks = len(split["new_order"])
    for task_index, subject in enumerate(split["new_order"], start=1):
        if task_index <= completed_tasks:
            continue
        fix_randomness(args.seed + 1000 * task_index)
        if args.reset_optimizer_per_subject and task_index > 1:
            optimizer = make_optimizer(student_blocks, args)
        subject = int(subject)
        clean_data_paths, clean_label_paths = subject_paths(args.data_root, subject)
        uploaded_paths, proxy_indices, proxy_diagnostics = resolve_uploaded_paths(
            args, task_index, subject, clean_data_paths
        )
        if len(uploaded_paths) != len(clean_data_paths):
            raise RuntimeError("Full PuriDivER input resolver violated N-to-N")
        clean_eval_loader = make_loader(
            args.data_root,
            [subject],
            args.eval_batch,
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
        guide_loader = make_unlabeled_loader(
            uploaded_paths,
            args,
            shuffle=True,
            seed=args.seed + 10_000 * task_index,
        )
        guide, cpc_losses = adapt_guiding_model(
            student_blocks, guide_loader, args, task_index, subject
        )
        pseudo_labels = infer_pseudo_labels(guide, uploaded_paths, args)
        diagnostic_loader = DataLoader(
            SequenceDataset((uploaded_paths, clean_label_paths)),
            batch_size=args.eval_batch,
            shuffle=False,
            num_workers=args.num_worker,
        )
        pseudo_diagnostics = pseudo_label_diagnostics(guide, diagnostic_loader, args)
        clean_pseudo_diagnostics = (
            pseudo_label_diagnostics(guide, clean_eval_loader, args)
            if proxy_indices
            else pseudo_diagnostics
        )
        del guide

        online = online_train_subject(
            student_blocks,
            optimizer,
            uploaded_paths,
            pseudo_labels,
            clean_data_paths,
            set(proxy_indices),
            memory,
            args,
            task_index,
            subject,
        )
        replay = replay_train(
            student_blocks, optimizer, memory, args, task_index
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
        seen_subjects.append(subject)
        performance["tasks"].append(
            {
                "task": task_index,
                "subject": subject,
                "current_before": before,
                "current_after": after,
                "old_generalization_after": old,
                "pseudo_labels": pseudo_diagnostics,
                "pseudo_labels_on_clean_current": clean_pseudo_diagnostics,
                "guiding_cpc_losses": cpc_losses,
                "proxy_upload": proxy_diagnostics,
                "online": online,
                "robust_replay": replay,
                "memory": memory.stats(),
                "training": {
                    "online_mini_batches": online["mini_batches"],
                    "memory_updates": online["memory_updates"],
                    "replay_epochs": len(replay),
                },
            }
        )
        if task_index in args.retention_milestones or task_index == total_tasks:
            performance["retention_snapshots"][str(task_index)] = (
                evaluate_seen_subjects(student_blocks, seen_subjects, args)
            )
        save_json(args.output_root / "metrics.json", performance)
        if not args.no_save_state:
            save_state(
                args.state_path,
                split=split,
                completed_tasks=task_index,
                blocks=student_blocks,
                optimizer=optimizer,
                memory=memory,
                performance=performance,
                manifest_sha256=manifest_sha256,
            )
        print(
            f"[full-puridiver] task={task_index}/{total_tasks} subject={subject} "
            f"current={before['acc']:.4f}->{after['acc']:.4f} "
            f"old={old['acc']:.4f} memory={len(memory)}",
            flush=True,
        )
        if args.stop_after_tasks and task_index >= args.stop_after_tasks:
            performance["interrupted_after_task"] = task_index
            save_json(args.output_root / "metrics.json", performance)
            return performance

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
    }
    performance["summary"] = summarize_run(performance)
    performance["summary"].update(
        {
            "final_memory_epochs": len(memory),
            "final_memory_proxy_epoch_fraction": memory.stats()[
                "proxy_epoch_fraction"
            ],
            "proxy_replay_fraction": memory.stats()["proxy_replay_fraction"],
        }
    )
    save_json(args.output_root / "metrics.json", performance)
    save_json(
        args.output_root / "summary.json",
        {"full_puridiver_eeg_adapted": performance["summary"]},
    )
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
        default=REPO_ROOT / "experiments" / "full_puridiver_eeg_runs" / "latest",
    )
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--online-batch-sequences", type=int, default=8)
    parser.add_argument("--replay-batch-sequences", type=int, default=8)
    parser.add_argument("--infer-batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--ssl-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssl-lr", type=float, default=1e-6)
    parser.add_argument("--cl-lr", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--memory-capacity-epochs", type=int, default=1000)
    parser.add_argument("--max-diversity-coefficient", type=float, default=0.5)
    parser.add_argument("--replay-epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-gmm-samples", type=int, default=8)
    parser.add_argument("--weak-noise", type=float, default=0.01)
    parser.add_argument("--weak-scale", type=float, default=0.02)
    parser.add_argument("--strong-noise", type=float, default=0.05)
    parser.add_argument("--strong-scale", type=float, default=0.10)
    parser.add_argument("--strong-mask-fraction", type=float, default=0.10)
    parser.add_argument("--strong-channel-drop", type=float, default=0.05)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--freeze-student-bn-stats", action="store_true")
    parser.add_argument(
        "--reset-optimizer-per-subject",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-online-batches", type=int, default=0)
    parser.add_argument("--max-replay-batches", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--retention-milestones", type=str, default="10,25,49")
    parser.add_argument("--n2n-manifest", type=Path, default=None)
    parser.add_argument(
        "--n2n-verify",
        choices=("none", "selected", "full"),
        default="selected",
    )
    parser.add_argument("--skip-n2n-prevalidation", action="store_true")
    parser.add_argument("--resume-state", type=Path, default=None)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--no-save-state", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int, default=0)
    args = parser.parse_args()

    if args.memory_capacity_epochs < 1:
        parser.error("PuriDivER memory capacity must be positive")
    if min(
        args.online_batch_sequences,
        args.replay_batch_sequences,
        args.infer_batch,
    ) < 1:
        parser.error("PuriDivER batch sizes must be positive")
    if args.replay_epochs < 1 or not 0 <= args.warmup_epochs <= args.replay_epochs:
        parser.error("PuriDivER warmup/replay epoch counts are invalid")
    if args.min_gmm_samples < 2:
        parser.error("--min-gmm-samples must be at least two")
    if not 0.0 <= args.max_diversity_coefficient <= 1.0:
        parser.error("--max-diversity-coefficient must be in [0, 1]")
    if args.stop_after_tasks < 0:
        parser.error("--stop-after-tasks cannot be negative")
    for name in (
        "weak_noise",
        "weak_scale",
        "strong_noise",
        "strong_scale",
        "strong_mask_fraction",
        "strong_channel_drop",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.n2n_manifest is not None:
        args.n2n_manifest = args.n2n_manifest.resolve()
        if not args.n2n_manifest.is_file():
            parser.error(f"Canonical N-to-N manifest does not exist: {args.n2n_manifest}")
    if args.resume_state is not None:
        args.resume_state = args.resume_state.resolve()
        if not args.resume_state.is_file():
            parser.error(f"Resume state does not exist: {args.resume_state}")
    args.dataset = "ISRUC"
    args.model_param = ModelConfig(args.dataset)
    args.device = torch.device(
        f"cuda:{args.gpu}"
        if args.gpu >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    args.retention_milestones = parse_int_set(args.retention_milestones)
    args.output_root = args.output_root.resolve()
    args.state_path = (
        args.output_root / "latest_state.pt"
        if args.state_path is None
        else args.state_path.resolve()
    )
    return args


if __name__ == "__main__":
    run(parse_args())
