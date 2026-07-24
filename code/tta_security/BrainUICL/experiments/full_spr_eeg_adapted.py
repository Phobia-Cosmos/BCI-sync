#!/usr/bin/env python3
"""Full unlabeled SPR lifecycle adapted to the BrainUICL EEG backbone."""

from __future__ import annotations

import argparse
import copy
import json
import math
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

from model.full_spr_eeg import (  # noqa: E402
    EEGContrastiveEncoder,
    SPRDelayedBuffer,
    SPRDelayedRecord,
    SPRPurifiedMemory,
    SPRPurifiedRecord,
    augment_eeg_views,
    self_centered_admission,
)
from model.regularization_cl import freeze_batch_norm_running_stats  # noqa: E402
from model.spr_eeg import NTXentLoss  # noqa: E402
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
    flat_labels,
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


STATE_SCHEMA = "full-spr-eeg-adapted-state-v1"


class MaskedPurifiedDataset(Dataset):
    """Load full sequence context and supervise only retained memory epochs."""

    def __init__(self, records: Sequence[SPRPurifiedRecord], sequence_length: int):
        self.records = list(records)
        self.sequence_length = int(sequence_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        values = np.load(record.data_path, allow_pickle=False).astype(
            np.float32, copy=False
        )
        if values.shape[0] != self.sequence_length:
            raise ValueError(f"Unexpected sequence length: {record.data_path}")
        labels = np.full(self.sequence_length, -100, dtype=np.int64)
        labels[record.epoch_mask] = record.pseudo_labels[record.epoch_mask]
        values = torch.from_numpy(values)
        return (
            values[:, :2, :],
            values[:, 2:, :],
            torch.from_numpy(labels),
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
    paths: Sequence[Path],
    args,
    *,
    shuffle: bool,
    seed: int,
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
    loader = make_unlabeled_loader(
        paths, args, shuffle=False, seed=args.seed
    )
    set_train(blocks, False)
    rows: list[np.ndarray] = []
    for eog, eeg, _dummy in loader:
        logits = forward_blocks(
            blocks, eog.to(args.device), eeg.to(args.device), args
        )
        rows.extend(logits.argmax(dim=1).detach().cpu().numpy())
    if len(rows) != len(paths):
        raise RuntimeError("Guide pseudo-label count does not match the upload stream")
    return [np.asarray(row, dtype=np.int64) for row in rows]


def make_contrastive_encoder(blocks, args) -> EEGContrastiveEncoder:
    return EEGContrastiveEncoder(
        copy.deepcopy(blocks[0]).to(args.device),
        copy.deepcopy(blocks[1]).to(args.device),
        embedding_dim=args.model_param.EncoderParam.d_model,
        projection_hidden=args.projection_hidden,
        projection_dim=args.projection_dim,
    ).to(args.device)


def _ntxent_step(model, optimizer, eog, eeg, criterion, args) -> float:
    first_eog, first_eeg = augment_eeg_views(
        eog,
        eeg,
        jitter=args.jitter,
        scale=args.augmentation_scale,
        mask_ratio=args.mask_ratio,
        channel_drop=args.channel_drop,
    )
    second_eog, second_eeg = augment_eeg_views(
        eog,
        eeg,
        jitter=args.jitter,
        scale=args.augmentation_scale,
        mask_ratio=args.mask_ratio,
        channel_drop=args.channel_drop,
    )
    first = model.projected_epochs(first_eog, first_eeg, args)
    second = model.projected_epochs(second_eog, second_eeg, args)
    if first.shape[0] < 2:
        raise RuntimeError("NT-Xent requires at least two logical EEG epochs")
    loss = criterion(first, second)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return float(loss.detach().cpu())


def train_expert(
    expert: EEGContrastiveEncoder,
    paths: Sequence[Path],
    args,
    flush_index: int,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(
        expert.parameters(), lr=args.spr_ssl_lr, weight_decay=args.weight_decay
    )
    criterion = NTXentLoss(args.temperature)
    losses: list[float] = []
    for epoch in range(args.expert_epochs):
        loader = make_unlabeled_loader(
            paths,
            args,
            shuffle=True,
            seed=args.seed + 100_000 * flush_index + epoch,
        )
        expert.train()
        if args.freeze_spr_bn_stats:
            freeze_batch_norm_running_stats(
                (expert.feature_extractor, expert.feature_encoder)
            )
        for batch_index, (eog, eeg, _dummy) in enumerate(loader):
            if args.max_ssl_batches and batch_index >= args.max_ssl_batches:
                break
            losses.append(
                _ntxent_step(
                    expert,
                    optimizer,
                    eog.to(args.device),
                    eeg.to(args.device),
                    criterion,
                    args,
                )
            )
    return {
        "epochs": args.expert_epochs,
        "steps": len(losses),
        "mean_ntxent": float(np.mean(losses)) if losses else math.nan,
    }


def _load_replay_records(
    records: Sequence[SPRPurifiedRecord], args
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not records:
        return None
    values = torch.stack(
        [
            torch.from_numpy(
                np.load(record.data_path, allow_pickle=False).astype(
                    np.float32, copy=False
                )
            )
            for record in records
        ]
    )
    return values[:, :, :2, :], values[:, :, 2:, :]


def train_base(
    base: EEGContrastiveEncoder,
    optimizer: torch.optim.Optimizer,
    current_paths: Sequence[Path],
    memory: SPRPurifiedMemory,
    args,
    flush_index: int,
) -> dict[str, Any]:
    criterion = NTXentLoss(args.temperature)
    losses: list[float] = []
    replay_sequences = 0
    for epoch in range(args.base_epochs):
        loader = make_unlabeled_loader(
            current_paths,
            args,
            shuffle=True,
            seed=args.seed + 200_000 * flush_index + epoch,
        )
        base.train()
        if args.freeze_spr_bn_stats:
            freeze_batch_norm_running_stats(
                (base.feature_extractor, base.feature_encoder)
            )
        for batch_index, (eog, eeg, _dummy) in enumerate(loader):
            if args.max_ssl_batches and batch_index >= args.max_ssl_batches:
                break
            replay_records = memory.sample_records(eog.shape[0])
            replay = _load_replay_records(replay_records, args)
            if replay is not None:
                replay_eog, replay_eeg = replay
                eog = torch.cat((eog, replay_eog), dim=0)
                eeg = torch.cat((eeg, replay_eeg), dim=0)
                replay_sequences += len(replay_records)
            losses.append(
                _ntxent_step(
                    base,
                    optimizer,
                    eog.to(args.device),
                    eeg.to(args.device),
                    criterion,
                    args,
                )
            )
    return {
        "epochs": args.base_epochs,
        "steps": len(losses),
        "mean_ntxent": float(np.mean(losses)) if losses else math.nan,
        "replay_sequences": replay_sequences,
    }


@torch.no_grad()
def collect_expert_embeddings(
    expert: EEGContrastiveEncoder,
    records: Sequence[SPRDelayedRecord],
    args,
) -> np.ndarray:
    loader = make_unlabeled_loader(
        [record.data_path for record in records],
        args,
        shuffle=False,
        seed=args.seed,
    )
    expert.eval()
    rows = []
    for eog, eeg, _dummy in loader:
        rows.append(
            expert.epoch_embeddings(
                eog.to(args.device), eeg.to(args.device), args
            ).cpu()
        )
    return torch.cat(rows, dim=0).numpy()


def finetune_inference(
    base: EEGContrastiveEncoder,
    previous_inference,
    memory: SPRPurifiedMemory,
    args,
    flush_index: int,
) -> tuple[tuple[torch.nn.Module, ...], dict[str, Any]]:
    if not memory.records:
        return previous_inference, {"epochs": 0, "steps": 0, "mean_ce": None}
    blocks = (
        copy.deepcopy(base.feature_extractor).to(args.device),
        copy.deepcopy(base.feature_encoder).to(args.device),
        copy.deepcopy(previous_inference[2]).to(args.device),
    )
    optimizer = torch.optim.Adam(
        [parameter for block in blocks for parameter in block.parameters()],
        lr=args.ft_lr,
        weight_decay=args.weight_decay,
    )
    losses: list[float] = []
    dataset = MaskedPurifiedDataset(memory.records, args.model_param.SeqLength)
    for epoch in range(args.ft_epochs):
        generator = torch.Generator().manual_seed(
            args.seed + 300_000 * flush_index + epoch
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_worker,
            generator=generator,
        )
        set_train(blocks, True)
        if args.freeze_spr_bn_stats:
            freeze_batch_norm_running_stats(blocks)
        for batch_index, (eog, eeg, labels) in enumerate(loader):
            if args.max_ft_batches and batch_index >= args.max_ft_batches:
                break
            logits = forward_blocks(
                blocks, eog.to(args.device), eeg.to(args.device), args
            )
            loss = F.cross_entropy(
                flat_logits(logits),
                flat_labels(labels.to(args.device)),
                ignore_index=-100,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for block in blocks for parameter in block.parameters()],
                    args.grad_clip,
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return blocks, {
        "epochs": args.ft_epochs,
        "steps": len(losses),
        "mean_ce": float(np.mean(losses)) if losses else math.nan,
        "supervised_retained_epochs": len(memory),
    }


def make_fresh_expert(initial_blocks, args, flush_index: int) -> EEGContrastiveEncoder:
    devices = [args.device.index or 0] if args.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.seed + 400_000 * flush_index)
        if args.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 400_000 * flush_index)
        return make_contrastive_encoder(initial_blocks, args)


def flush_spr(
    *,
    delayed_buffer: SPRDelayedBuffer,
    memory: SPRPurifiedMemory,
    base: EEGContrastiveEncoder,
    base_optimizer: torch.optim.Optimizer,
    inference_blocks,
    initial_blocks,
    admission_rng: np.random.Generator,
    args,
    flush_index: int,
    trigger: str,
) -> tuple[tuple[torch.nn.Module, ...], dict[str, Any]]:
    delayed = delayed_buffer.drain()
    if not delayed:
        raise ValueError("Cannot flush an empty SPR Delayed Buffer")
    paths = [record.data_path for record in delayed]
    expert = make_fresh_expert(initial_blocks, args, flush_index)
    expert_training = train_expert(expert, paths, args, flush_index)
    base_training = train_base(
        base, base_optimizer, paths, memory, args, flush_index
    )
    features = collect_expert_embeddings(expert, delayed, args)
    masks, probabilities, admission = self_centered_admission(
        features,
        delayed,
        ensembles=args.spr_ensembles,
        bmm_iters=args.spr_bmm_iters,
        graph_seed=args.seed + 500_000 * flush_index,
        admission_rng=admission_rng,
    )
    memory_update = memory.add(delayed, masks, probabilities)
    inference_blocks, finetune = finetune_inference(
        base, inference_blocks, memory, args, flush_index
    )
    del expert
    return inference_blocks, {
        "flush": flush_index,
        "trigger": trigger,
        "expert_reset": True,
        "expert_training": expert_training,
        "base_self_replay": base_training,
        "admission": admission,
        "memory_update": memory_update,
        "memory_after": memory.stats(),
        "inference_finetune": finetune,
    }


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
    flush_index: int,
    base: EEGContrastiveEncoder,
    base_optimizer: torch.optim.Optimizer,
    inference_blocks,
    delayed_buffer: SPRDelayedBuffer,
    memory: SPRPurifiedMemory,
    admission_rng: np.random.Generator,
    performance: dict,
    manifest_sha256: str | None,
) -> None:
    payload = {
        "schema": STATE_SCHEMA,
        "split": split,
        "completed_tasks": int(completed_tasks),
        "flush_index": int(flush_index),
        "base": base.state_dict(),
        "base_optimizer": base_optimizer.state_dict(),
        "inference_blocks": [block.state_dict() for block in inference_blocks],
        "delayed_buffer": delayed_buffer.state_dict(),
        "purified_memory": memory.state_dict(),
        "admission_rng_state": admission_rng.bit_generator.state,
        "rng": _capture_rng_state(),
        "performance": performance,
        "manifest_sha256": manifest_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_state(
    path: Path,
    *,
    split: dict,
    base: EEGContrastiveEncoder,
    base_optimizer: torch.optim.Optimizer,
    inference_blocks,
    args,
    manifest_sha256: str | None,
) -> tuple[int, int, SPRDelayedBuffer, SPRPurifiedMemory, np.random.Generator, dict]:
    payload = torch.load(path, map_location=args.device, weights_only=False)
    if payload.get("schema") != STATE_SCHEMA:
        raise ValueError(f"Unsupported Full SPR state: {payload.get('schema')}")
    if payload["split"]["new_order"] != split["new_order"]:
        raise ValueError("Full SPR resume split/order differs from the current run")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Full SPR resume manifest differs from the current run")
    base.load_state_dict(payload["base"])
    base_optimizer.load_state_dict(payload["base_optimizer"])
    for block, state in zip(inference_blocks, payload["inference_blocks"]):
        block.load_state_dict(state)
    delayed_buffer = SPRDelayedBuffer.from_state_dict(payload["delayed_buffer"])
    memory = SPRPurifiedMemory.from_state_dict(payload["purified_memory"])
    admission_rng = np.random.default_rng()
    admission_rng.bit_generator.state = payload["admission_rng_state"]
    _restore_rng_state(payload["rng"])
    return (
        int(payload["completed_tasks"]),
        int(payload["flush_index"]),
        delayed_buffer,
        memory,
        admission_rng,
        payload["performance"],
    )


def resolve_uploaded_paths(
    args,
    task_index: int,
    subject: int,
    clean_data_paths: Sequence[Path],
) -> tuple[list[Path], tuple[int, ...], dict | None]:
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
    inference_blocks = tuple(copy.deepcopy(block).to(args.device) for block in initial_blocks)
    base = make_contrastive_encoder(initial_blocks, args)
    base_optimizer = torch.optim.Adam(
        base.parameters(), lr=args.spr_ssl_lr, weight_decay=args.weight_decay
    )
    delayed_buffer = SPRDelayedBuffer(args.delayed_capacity_sequences)
    memory = SPRPurifiedMemory(
        args.memory_capacity_epochs, args.model_param.NumClasses, args.seed + 17
    )
    admission_rng = np.random.default_rng(args.seed + 23)
    completed_tasks = 0
    flush_index = 0

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
            inference_blocks,
            old_loader,
            args,
            max_batches=args.eval_max_batches,
        )
    )
    performance = {
        "method": "full_spr_eeg_adapted",
        "protocol": {
            "backbone": "BrainUICL FeatureExtractor + TransformerEncoder + SleepMLP",
            "initialization": "common source-supervised checkpoint",
            "target_training_loader": "signal-only without annotation paths",
            "pseudo_labels": "task-local CPC guide hard argmax for every epoch",
            "confidence_filter": False,
            "delayed_buffer_capacity_sequences": args.delayed_capacity_sequences,
            "purified_memory_capacity_epochs": args.memory_capacity_epochs,
            "expert_reset_per_flush": True,
            "expert_object_shared_with_base": False,
            "expert_object_shared_with_inference": False,
            "expert_objective": "NT-Xent on D",
            "base_objective": "persistent NT-Xent Self-Replay on D plus P",
            "admission": "class-conditional stochastic SCF with Beta mixture and Bernoulli",
            "inference_update": "masked pseudo-label CE on retained P epochs only",
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
                    inference_blocks,
                    source_loader,
                    args,
                    max_batches=args.eval_max_batches,
                )
            ),
            "validation": metric_view(
                evaluate(
                    inference_blocks,
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
        (
            completed_tasks,
            flush_index,
            delayed_buffer,
            memory,
            admission_rng,
            performance,
        ) = load_state(
            args.resume_state,
            split=split,
            base=base,
            base_optimizer=base_optimizer,
            inference_blocks=inference_blocks,
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
        subject = int(subject)
        clean_data_paths, clean_label_paths = subject_paths(args.data_root, subject)
        uploaded_paths, proxy_indices, proxy_diagnostics = resolve_uploaded_paths(
            args, task_index, subject, clean_data_paths
        )
        if len(uploaded_paths) != len(clean_data_paths):
            raise RuntimeError("Full SPR input resolver violated N-to-N cardinality")
        proxy_set = set(proxy_indices)
        clean_eval_loader = make_loader(
            args.data_root,
            [subject],
            args.eval_batch,
            shuffle=False,
            num_workers=args.num_worker,
        )
        before = metric_view(
            evaluate(
                inference_blocks,
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
            inference_blocks, guide_loader, args, task_index, subject
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

        flush_rows: list[dict[str, Any]] = []
        for slot, (data_path, labels) in enumerate(zip(uploaded_paths, pseudo_labels)):
            became_full = delayed_buffer.add(
                SPRDelayedRecord(
                    data_path=data_path,
                    pseudo_labels=labels,
                    task=task_index,
                    subject=subject,
                    sequence_index=int(clean_data_paths[slot].stem),
                    is_proxy=slot in proxy_set,
                )
            )
            if became_full:
                flush_index += 1
                inference_blocks, flush_row = flush_spr(
                    delayed_buffer=delayed_buffer,
                    memory=memory,
                    base=base,
                    base_optimizer=base_optimizer,
                    inference_blocks=inference_blocks,
                    initial_blocks=initial_blocks,
                    admission_rng=admission_rng,
                    args=args,
                    flush_index=flush_index,
                    trigger="capacity",
                )
                flush_rows.append(flush_row)
        if len(delayed_buffer):
            flush_index += 1
            inference_blocks, flush_row = flush_spr(
                delayed_buffer=delayed_buffer,
                memory=memory,
                base=base,
                base_optimizer=base_optimizer,
                inference_blocks=inference_blocks,
                initial_blocks=initial_blocks,
                admission_rng=admission_rng,
                args=args,
                flush_index=flush_index,
                trigger="subject_end_residual",
            )
            flush_rows.append(flush_row)

        after = metric_view(
            evaluate(
                inference_blocks,
                clean_eval_loader,
                args,
                max_batches=args.eval_max_batches,
            )
        )
        old = metric_view(
            evaluate(
                inference_blocks,
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
                "flushes": flush_rows,
                "memory": memory.stats(),
                "training": {
                    "flush_count": len(flush_rows),
                    "expert_resets": len(flush_rows),
                    "delayed_buffer_empty_after_subject": len(delayed_buffer) == 0,
                },
            }
        )
        if task_index in args.retention_milestones or task_index == total_tasks:
            performance["retention_snapshots"][str(task_index)] = (
                evaluate_seen_subjects(inference_blocks, seen_subjects, args)
            )
        save_json(args.output_root / "metrics.json", performance)
        if not args.no_save_state:
            save_state(
                args.state_path,
                split=split,
                completed_tasks=task_index,
                flush_index=flush_index,
                base=base,
                base_optimizer=base_optimizer,
                inference_blocks=inference_blocks,
                delayed_buffer=delayed_buffer,
                memory=memory,
                admission_rng=admission_rng,
                performance=performance,
                manifest_sha256=manifest_sha256,
            )
        print(
            f"[full-spr] task={task_index}/{total_tasks} subject={subject} "
            f"current={before['acc']:.4f}->{after['acc']:.4f} "
            f"old={old['acc']:.4f} P={len(memory)} flushes={len(flush_rows)}",
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
                inference_blocks,
                old_loader,
                args,
                max_batches=args.eval_max_batches,
            )
        ),
        "source_train": metric_view(
            evaluate(
                inference_blocks,
                source_loader,
                args,
                max_batches=args.eval_max_batches,
            )
        ),
        "validation": metric_view(
            evaluate(
                inference_blocks,
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
            "total_flushes": flush_index,
        }
    )
    save_json(args.output_root / "metrics.json", performance)
    save_json(
        args.output_root / "summary.json",
        {"full_spr_eeg_adapted": performance["summary"]},
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
        default=REPO_ROOT / "experiments" / "full_spr_eeg_runs" / "latest",
    )
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--ssl-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssl-lr", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--delayed-capacity-sequences", type=int, default=32)
    parser.add_argument("--memory-capacity-epochs", type=int, default=1000)
    parser.add_argument("--expert-epochs", type=int, default=10)
    parser.add_argument("--base-epochs", type=int, default=10)
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--spr-ssl-lr", type=float, default=1e-6)
    parser.add_argument("--ft-lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--projection-hidden", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--jitter", type=float, default=0.02)
    parser.add_argument("--augmentation-scale", type=float, default=0.10)
    parser.add_argument("--mask-ratio", type=float, default=0.05)
    parser.add_argument("--channel-drop", type=float, default=0.05)
    parser.add_argument("--spr-ensembles", type=int, default=5)
    parser.add_argument("--spr-bmm-iters", type=int, default=10)
    parser.add_argument("--freeze-spr-bn-stats", action="store_true")
    parser.add_argument("--max-ssl-batches", type=int, default=0)
    parser.add_argument("--max-ft-batches", type=int, default=0)
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
    parser.add_argument(
        "--stop-after-tasks",
        type=int,
        default=0,
        help="Stop at a task boundary after saving state; used for resume validation.",
    )
    args = parser.parse_args()

    if args.delayed_capacity_sequences < 1 or args.memory_capacity_epochs < 1:
        parser.error("SPR buffer capacities must be positive")
    if args.stop_after_tasks < 0:
        parser.error("--stop-after-tasks cannot be negative")
    if min(args.expert_epochs, args.base_epochs, args.ft_epochs) < 1:
        parser.error("Full SPR training epochs must be positive")
    if args.temperature <= 0 or args.spr_ssl_lr <= 0 or args.ft_lr <= 0:
        parser.error("Full SPR learning rates and temperature must be positive")
    if args.spr_ensembles < 1 or args.spr_bmm_iters < 1:
        parser.error("SPR SCF iteration counts must be positive")
    for name in ("jitter", "augmentation_scale", "mask_ratio", "channel_drop"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
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
