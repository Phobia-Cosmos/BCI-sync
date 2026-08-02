#!/usr/bin/env python3
"""Generate label-free individual-order curricula for a fixed CL partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from progressive_feedback_proxy import (  # noqa: E402
    physiological_eeg_descriptor,
    public_probabilities,
)
from rttdp_brainuicl_full import (  # noqa: E402
    discover_subjects,
    load_pretrained,
    split_subjects,
    subject_paths,
)
from utils.config import ModelConfig  # noqa: E402
from utils.util import fix_randomness  # noqa: E402


ORDER_NAMES = (
    "seed_random",
    "easy_to_hard",
    "hard_to_easy",
    "source_near_to_far",
    "smooth_nearest",
    "diversity_greedy",
)


def sample_paths(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def subject_descriptor(data_root: Path, subject: int, dataset: str, count: int) -> np.ndarray:
    paths = sample_paths(subject_paths(data_root, subject)[0], count)
    return np.mean([
        physiological_eeg_descriptor(np.load(path, allow_pickle=False), dataset)
        for path in paths
    ], axis=0)


def greedy_path(descriptors: np.ndarray, start: int, *, farthest: bool) -> list[int]:
    selected = [int(start)]
    remaining = set(range(len(descriptors))) - {int(start)}
    while remaining:
        if farthest:
            index = max(
                remaining,
                key=lambda candidate: min(
                    float(np.linalg.norm(descriptors[candidate] - descriptors[past]))
                    for past in selected
                ),
            )
        else:
            index = min(
                remaining,
                key=lambda candidate: float(
                    np.linalg.norm(descriptors[candidate] - descriptors[selected[-1]])
                ),
            )
        selected.append(int(index))
        remaining.remove(index)
    return selected


def order_diagnostics(order: list[int], subject_rows: dict[int, dict]) -> dict:
    descriptors = np.stack([subject_rows[subject]["standardized_descriptor"] for subject in order])
    steps = np.linalg.norm(np.diff(descriptors, axis=0), axis=1)
    entropy = np.asarray([subject_rows[subject]["entropy"] for subject in order])
    positions = np.arange(len(order), dtype=np.float64)
    return {
        "path_length": float(steps.sum()),
        "mean_step_distance": float(steps.mean()),
        "max_step_distance": float(steps.max()),
        "entropy_start_quartile": float(entropy[: max(1, len(order) // 4)].mean()),
        "entropy_end_quartile": float(entropy[-max(1, len(order) // 4) :].mean()),
        "entropy_position_correlation": float(np.corrcoef(positions, entropy)[0, 1]),
    }


def generate(args) -> dict:
    fix_randomness(args.seed)
    subjects = discover_subjects(args.data_root)
    train_idx, val_idx, old_idx, new_idx = split_subjects(subjects, args.seed)
    model_args = SimpleNamespace(
        dataset=args.dataset,
        data_root=args.data_root,
        input_checkpoint_root=args.input_checkpoint_root,
        seed=args.seed,
        pretrain_seed=args.pretrain_seed,
        batch=args.batch,
        num_worker=0,
        device=torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"),
        model_param=ModelConfig(args.dataset),
    )
    blocks = load_pretrained(model_args)
    source_descriptors = np.stack([
        subject_descriptor(args.data_root, int(subject), args.dataset, args.descriptor_sequences)
        for subject in train_idx
    ])
    rows: dict[int, dict] = {}
    for subject_value in new_idx:
        subject = int(subject_value)
        paths = subject_paths(args.data_root, subject)[0]
        descriptor = subject_descriptor(
            args.data_root, subject, args.dataset, args.descriptor_sequences
        )
        probabilities = public_probabilities(
            blocks, sample_paths(paths, args.feedback_sequences), model_args
        ).reshape(-1, model_args.model_param.NumClasses)
        entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-8)), axis=1)
        rows[subject] = {
            "subject": subject,
            "sequences": len(paths),
            "descriptor": descriptor.tolist(),
            "entropy": float(entropy.mean()),
            "confidence": float(probabilities.max(axis=1).mean()),
            "predicted_distribution": probabilities.mean(axis=0).tolist(),
        }
    new_subjects = [int(subject) for subject in new_idx]
    matrix = np.stack([rows[subject]["descriptor"] for subject in new_subjects])
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 1e-8)
    standardized = (matrix - mean) / scale
    source_centroid = (source_descriptors.mean(axis=0) - mean) / scale
    for index, subject in enumerate(new_subjects):
        rows[subject]["standardized_descriptor"] = standardized[index].tolist()
        rows[subject]["source_distance"] = float(
            np.linalg.norm(standardized[index] - source_centroid)
        )

    entropy = np.asarray([rows[subject]["entropy"] for subject in new_subjects])
    source_distance = np.asarray([rows[subject]["source_distance"] for subject in new_subjects])
    start_near = int(np.argmin(source_distance))
    start_far = int(np.argmax(source_distance))
    index_orders = {
        "seed_random": list(range(len(new_subjects))),
        "easy_to_hard": np.argsort(entropy, kind="stable").tolist(),
        "hard_to_easy": np.argsort(-entropy, kind="stable").tolist(),
        "source_near_to_far": np.argsort(source_distance, kind="stable").tolist(),
        "smooth_nearest": greedy_path(standardized, start_near, farthest=False),
        "diversity_greedy": greedy_path(standardized, start_far, farthest=True),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name, indices in index_orders.items():
        order = [new_subjects[index] for index in indices]
        manifest = {
            "dataset": args.dataset,
            "partition_seed": args.seed,
            "pretrain_seed": args.pretrain_seed,
            "order_name": name,
            "selection_information": "unlabeled signal statistics and public pretrained probabilities only",
            "true_incremental_labels_used": False,
            "new_order": order,
            "diagnostics": order_diagnostics(order, rows),
        }
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(manifest, indent=2))
        manifests[name] = str(path)
    output = {
        "dataset": args.dataset,
        "partition_seed": args.seed,
        "train_idx": [int(value) for value in train_idx],
        "val_idx": [int(value) for value in val_idx],
        "old_idx": [int(value) for value in old_idx],
        "new_subjects": new_subjects,
        "descriptor": {
            "relative_bandpower_hz": [[0.5, 4], [4, 8], [8, 13], [13, 30], [30, 45]],
            "normalized_channel_covariance_eigenvalues": 6,
            "autocorrelation_seconds": [0.01, 0.04, 0.16],
            "dataset_sampling_rate": 250 if args.dataset == "FACED" else 100,
            "eeg_channels_only": True,
        },
        "subjects": [rows[subject] for subject in new_subjects],
        "manifests": manifests,
    }
    (args.output_dir / "ANALYSIS.json").write_text(json.dumps(output, indent=2))
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ISRUC", "FACED"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--pretrain-seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--descriptor-sequences", type=int, default=4)
    parser.add_argument("--feedback-sequences", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    result = generate(parse_args())
    print(json.dumps({"dataset": result["dataset"], "manifests": result["manifests"]}))
