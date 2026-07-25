#!/usr/bin/env python3
"""Generate one immutable partial N-to-N proxy stream from a clean Finetune surrogate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.regularization_cl import build_regularization_strategy  # noqa: E402
from n2n_shared_proxy import (  # noqa: E402
    load_manifest,
    make_task_entry,
    sha256_file,
    signal_delta_metadata,
    validate_manifest,
    write_manifest,
)
from regularization_cl_attacks import (  # noqa: E402
    materialize_batched_proxy_dual_harm_subject,
)
from regularization_cl_eeg import (  # noqa: E402
    adapt_guiding_model,
    build_split,
    make_unlabeled_loader,
    train_student_task,
)
from rttdp_brainuicl_full import load_pretrained  # noqa: E402
from utils.config import ModelConfig  # noqa: E402
from utils.util import fix_randomness  # noqa: E402


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_affected_tasks(value: str, fallback: int) -> tuple[int, ...]:
    tasks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not tasks:
        tasks = [int(fallback)]
    if any(task < 1 for task in tasks):
        raise ValueError("Affected tasks must be positive")
    return tuple(sorted(set(tasks)))


def signal_paths(data_root: Path, subject: int) -> list[Path]:
    data_dir = data_root / str(subject) / "data"
    paths: list[Path] = []
    index = 0
    while (data_dir / f"{index}.npy").is_file():
        paths.append((data_dir / f"{index}.npy").resolve())
        index += 1
    if not paths:
        raise FileNotFoundError(f"No signal sequences for subject {subject}: {data_dir}")
    return paths


def merge_signal_paths(data_root: Path, subjects: Sequence[int]) -> list[Path]:
    return [
        path
        for subject in subjects
        for path in signal_paths(data_root, int(subject))
    ]


def load_surrogate(args):
    blocks = load_pretrained(args)
    if args.surrogate_resume_task == 0:
        return blocks, {}
    names = ("feature_extractor", "feature_encoder", "sleep_classifier")
    hashes: dict[str, str] = {}
    for block, name in zip(blocks, names):
        path = args.surrogate_checkpoint_dir / f"{name}_parameter_{args.seed}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing surrogate checkpoint block: {path}")
        block.load_state_dict(torch.load(path, map_location=args.device))
        hashes[name] = sha256_file(path)
    return blocks, hashes


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def enforce_payload_budget(
    clean_path: Path,
    proxy_path: Path,
    *,
    max_relative_l2: float,
    max_linf_over_std: float,
) -> dict[str, float]:
    clean = np.load(clean_path, allow_pickle=False).astype(np.float32, copy=False)
    proxy = np.load(proxy_path, allow_pickle=False).astype(np.float32, copy=False)
    delta = proxy - clean
    scale = 1.0
    for channel_slice in (slice(None), slice(0, 2), slice(2, None)):
        local_clean = clean[:, channel_slice]
        local_delta = delta[:, channel_slice]
        if local_clean.size == 0:
            continue
        delta_norm = float(np.linalg.norm(local_delta.astype(np.float64).reshape(-1)))
        clean_norm = float(np.linalg.norm(local_clean.astype(np.float64).reshape(-1)))
        if delta_norm > 0 and max_relative_l2 > 0:
            scale = min(scale, max_relative_l2 * clean_norm / delta_norm)
        delta_max = float(np.abs(local_delta.astype(np.float64)).max(initial=0.0))
        clean_std = float(local_clean.astype(np.float64).std())
        if delta_max > 0 and max_linf_over_std > 0:
            scale = min(scale, max_linf_over_std * clean_std / delta_max)
    if scale < 1.0:
        # Leave room for float32 subtraction/serialization before manifest
        # validation recomputes the ratio in float64.
        proxy = clean + delta * np.float32(max(scale * (1.0 - 1e-4), 0.0))
        np.save(proxy_path, proxy.astype(np.float32, copy=False))
    return signal_delta_metadata(clean_path, proxy_path)


def tensor_state_sha256(blocks) -> str:
    digest = hashlib.sha256()
    for block in blocks:
        for name, value in sorted(block.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    split = build_split(args)
    affected_tasks = tuple(
        getattr(args, "affected_tasks", ()) or (int(args.affected_task),)
    )
    expected = {
        task: (int(subject), signal_paths(args.data_root, int(subject)))
        for task, subject in enumerate(split["new_order"], start=1)
    }
    if manifest_path.is_file() and not args.regenerate:
        existing = load_manifest(manifest_path)
        constraints = existing.get("constraints", {})
        expected_constraints = {
            "affected_tasks": list(affected_tasks),
            "sequence_fraction": float(args.attack_fraction),
            "max_relative_l2": float(args.attack_max_relative_l2),
            "max_linf_over_std": float(args.attack_eps_scale),
            "repeat": 0,
        }
        for name, expected_value in expected_constraints.items():
            actual_value = constraints.get(name)
            if isinstance(expected_value, float):
                matches = actual_value is not None and math.isclose(
                    float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12
                )
            else:
                matches = actual_value == expected_value
            if not matches:
                raise ValueError(
                    f"Existing manifest constraint {name}={actual_value!r} does not "
                    f"match requested {expected_value!r}"
                )
        report = validate_manifest(manifest_path, expected)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return report
    total_tasks = len(split["new_order"])
    invalid_tasks = [
        task for task in affected_tasks if not 1 <= int(task) <= total_tasks
    ]
    if invalid_tasks:
        raise ValueError(
            f"Affected tasks must be in 1..{total_tasks}: {invalid_tasks}"
        )
    if args.surrogate_resume_task >= min(affected_tasks):
        raise ValueError("Surrogate checkpoint must precede every affected task")

    fix_randomness(args.seed)
    surrogate, checkpoint_hashes = load_surrogate(args)
    strategy = build_regularization_strategy(
        "finetune",
        ewc_strength=0.0,
        online_ewc_strength=0.0,
        online_ewc_decay=1.0,
        si_strength=0.0,
        si_xi=0.1,
        mas_strength=0.0,
        mas_decay=1.0,
    )
    source_reference_paths = merge_signal_paths(args.data_root, split["train_idx"])
    proxy_paths_by_task: dict[int, dict[int, Path]] = {}
    generations: dict[int, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []

    for task_index, subject_value in enumerate(split["new_order"], start=1):
        if task_index <= args.surrogate_resume_task:
            continue
        subject = int(subject_value)
        fix_randomness(args.seed + 1000 * task_index)
        clean_paths = signal_paths(args.data_root, subject)
        clean_loader = make_unlabeled_loader(args, clean_paths, True)
        guide, cpc_losses = adapt_guiding_model(
            surrogate, clean_loader, args, task_index, subject
        )
        if task_index in affected_tasks:
            rng_state = capture_rng()
            mixed_paths, generation = materialize_batched_proxy_dual_harm_subject(
                student_blocks=surrogate,
                label_blocks=guide,
                strategy=None,
                current_data_paths=clean_paths,
                reference_data_paths=source_reference_paths,
                output_dir=args.output_root / "payload" / f"task_{task_index}",
                task_index=task_index,
                subject=subject,
                args=args,
            )
            restore_rng(rng_state)
            task_proxy_paths: dict[int, Path] = {}
            for slot in generation["poison_indices"]:
                clean_path = clean_paths[int(slot)]
                proxy_path = Path(mixed_paths[int(slot)]).resolve()
                enforce_payload_budget(
                    clean_path,
                    proxy_path,
                    max_relative_l2=args.attack_max_relative_l2,
                    max_linf_over_std=args.attack_eps_scale,
                )
                task_proxy_paths[int(slot)] = proxy_path
            proxy_paths_by_task[task_index] = task_proxy_paths
            generations[task_index] = generation
        training, _importance, _curvature = train_student_task(
            surrogate,
            guide,
            clean_loader,
            strategy,
            args,
            task_index,
            subject,
        )
        trace.append(
            {
                "task": task_index,
                "subject": subject,
                "guiding_cpc_losses": cpc_losses,
                "last_clean_pseudo_loss": training["last_pseudo_loss"],
                "used_proxy_for_surrogate_update": False,
            }
        )
        save_json(args.output_root / "surrogate_trace.json", trace)

    if set(generations) != set(affected_tasks):
        raise RuntimeError(
            f"Proxy payloads were not generated for every task: "
            f"expected={list(affected_tasks)}, generated={sorted(generations)}"
        )
    entries = []
    for task_index, subject_value in enumerate(split["new_order"], start=1):
        entries.append(
            make_task_entry(
                task=task_index,
                subject=int(subject_value),
                clean_paths=expected[task_index][1],
                proxy_paths=proxy_paths_by_task.get(task_index, {}),
            )
        )
    payload = write_manifest(
        manifest_path,
        tasks=entries,
        split=split,
        constraints={
            "affected_tasks": list(affected_tasks),
            "upload_multiplier": 1,
            "repeat": 0,
            "sequence_fraction": args.attack_fraction,
            "max_relative_l2": args.attack_max_relative_l2,
            "max_linf_over_std": args.attack_eps_scale,
            "selection_seed": args.seed,
            "target_labels_available_to_generator": False,
        },
        provenance={
            "generator": "independent clean Finetune surrogate",
            "surrogate_resume_task": args.surrogate_resume_task,
            "surrogate_checkpoint_dir": str(args.surrogate_checkpoint_dir),
            "surrogate_checkpoint_sha256": checkpoint_hashes,
            "surrogate_continued_on_clean_through_task": total_tasks,
            "surrogate_final_state_sha256": tensor_state_sha256(surrogate),
            "proxy_objective": "dual old/new degradation through one-step classifier update",
            "victim_state_used": False,
            "target_annotations_opened": False,
            "generation_diagnostics": (
                generations[affected_tasks[0]]
                if len(affected_tasks) == 1
                else [generations[task] for task in affected_tasks]
            ),
        },
    )
    validation = validate_manifest(manifest_path, expected)
    summary = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "affected_tasks": list(affected_tasks),
        "affected_subjects": [
            int(split["new_order"][task - 1]) for task in affected_tasks
        ],
        "proxy_sequences": validation["proxy_sequences"],
        "uploaded_sequences": validation["uploaded_sequences"],
        "generations": [generations[task] for task in affected_tasks],
        "validation": validation,
        "schema": payload["schema"],
    }
    if len(affected_tasks) == 1:
        task = affected_tasks[0]
        summary.update(
            {
                "affected_task": task,
                "affected_subject": int(split["new_order"][task - 1]),
                "generation": generations[task],
            }
        )
    save_json(args.output_root / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32"),
    )
    parser.add_argument(
        "--input-checkpoint-root",
        type=Path,
        default=Path("/home/undefined/Disk/ai-storage/BrainUICL/model_parameter"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "canonical_n2n_shared_proxy" / "payload",
    )
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--affected-task", type=int, default=26)
    parser.add_argument(
        "--affected-tasks",
        type=str,
        default="",
        help="Optional comma-separated task list; overrides --affected-task.",
    )
    parser.add_argument("--surrogate-resume-task", type=int, default=25)
    parser.add_argument(
        "--surrogate-checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "regularization_cl_eeg_runs"
        / "clean49_bn_frozen_e10_lr1e6_seed4321" / "finetune" / "checkpoints"
        / "individual_25",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--ssl-epoch", type=int, default=10)
    parser.add_argument("--incremental-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssl-lr", type=float, default=1e-6)
    parser.add_argument("--cl-lr", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--freeze-bn-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack-mode", default="proxy_dual_harm")
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--attack-eps-scale", type=float, default=0.50)
    parser.add_argument("--attack-max-relative-l2", type=float, default=0.20)
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
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    try:
        args.affected_tasks = parse_affected_tasks(
            args.affected_tasks,
            args.affected_task,
        )
    except ValueError as error:
        parser.error(str(error))
    if not 0.0 < args.attack_fraction <= 1.0:
        parser.error("--attack-fraction must be in (0, 1]")
    if args.attack_max_relative_l2 <= 0 or args.attack_eps_scale <= 0:
        parser.error("Proxy budgets must be positive")
    args.dataset = "ISRUC"
    args.model_param = ModelConfig(args.dataset)
    args.device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    args.output_root = args.output_root.resolve()
    args.surrogate_checkpoint_dir = args.surrogate_checkpoint_dir.resolve()
    return args


if __name__ == "__main__":
    run(parse_args())
