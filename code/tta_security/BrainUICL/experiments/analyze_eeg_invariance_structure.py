#!/usr/bin/env python3
"""Compare within-subject, between-subject and cross-dataset EEG descriptors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from progressive_feedback_proxy import physiological_eeg_descriptor  # noqa: E402
from rttdp_brainuicl_full import discover_subjects, split_subjects, subject_paths  # noqa: E402
from utils.util import fix_randomness  # noqa: E402


COMPONENTS = (
    "delta_relative_power", "theta_relative_power", "alpha_relative_power",
    "beta_relative_power", "gamma_relative_power", "covariance_eigen_1",
    "covariance_eigen_2", "covariance_eigen_3", "covariance_eigen_4",
    "covariance_eigen_5", "covariance_eigen_6", "autocorrelation_10ms",
    "autocorrelation_40ms", "autocorrelation_160ms",
)


def sample_paths(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def dataset_statistics(dataset: str, root: Path, seed: int, sequences: int) -> dict:
    fix_randomness(seed)
    subjects = discover_subjects(root)
    train, _val, _old, new = split_subjects(subjects, seed)
    selected_subjects = [int(value) for value in [*train, *new]]
    subject_means, subject_variances = [], []
    for subject in selected_subjects:
        descriptors = np.stack([
            physiological_eeg_descriptor(np.load(path, allow_pickle=False), dataset)
            for path in sample_paths(subject_paths(root, subject)[0], sequences)
        ])
        subject_means.append(descriptors.mean(axis=0))
        subject_variances.append(descriptors.var(axis=0))
    subject_means = np.stack(subject_means)
    within_variance = np.stack(subject_variances).mean(axis=0)
    between_variance = subject_means.var(axis=0)
    mean = subject_means.mean(axis=0)
    scale = np.maximum(np.abs(mean), 0.05)
    normalized_total_std = np.sqrt(within_variance + between_variance) / scale
    rows = []
    for index, name in enumerate(COMPONENTS):
        rows.append({
            "component": name,
            "mean": float(mean[index]),
            "within_subject_std": float(np.sqrt(within_variance[index])),
            "between_subject_std": float(np.sqrt(between_variance[index])),
            "normalized_total_std": float(normalized_total_std[index]),
            "between_to_within_variance": float(
                between_variance[index] / max(within_variance[index], 1e-12)
            ),
        })
    return {
        "dataset": dataset,
        "sampling_rate": 250 if dataset == "FACED" else 100,
        "subjects": len(selected_subjects),
        "sequences_per_subject": sequences,
        "components": rows,
        "subject_mean_matrix": subject_means.tolist(),
    }


def cross_dataset_rows(first: dict, second: dict) -> list[dict]:
    output = []
    first_matrix = np.asarray(first["subject_mean_matrix"])
    second_matrix = np.asarray(second["subject_mean_matrix"])
    for index, name in enumerate(COMPONENTS):
        pooled = np.sqrt((first_matrix[:, index].var() + second_matrix[:, index].var()) / 2.0)
        output.append({
            "component": name,
            "standardized_dataset_mean_difference": float(
                abs(first_matrix[:, index].mean() - second_matrix[:, index].mean())
                / max(pooled, 1e-12)
            ),
            "numerically_similar_below_0_5": bool(
                abs(first_matrix[:, index].mean() - second_matrix[:, index].mean())
                / max(pooled, 1e-12) < 0.5
            ),
        })
    return output


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# EEG Invariance Structure Audit", "",
        "Lower normalized total standard deviation means the normalized statistic is more stable across sampled sequences and subjects. Cross-dataset similarity uses a standardized mean difference below 0.5 only as a screening rule, not proof of physiological equivalence.", "",
        "| Component | ISRUC normalized std | FACED normalized std | Cross-dataset effect size | Similar screen |",
        "|---|---:|---:|---:|---:|",
    ]
    first = {row["component"]: row for row in result["datasets"]["ISRUC"]["components"]}
    second = {row["component"]: row for row in result["datasets"]["FACED"]["components"]}
    cross = {row["component"]: row for row in result["cross_dataset"]}
    for name in COMPONENTS:
        lines.append(
            f"| {name} | {first[name]['normalized_total_std']:.4f} | "
            f"{second[name]['normalized_total_std']:.4f} | "
            f"{cross[name]['standardized_dataset_mean_difference']:.4f} | "
            f"{'yes' if cross[name]['numerically_similar_below_0_5'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isruc-root", type=Path, required=True)
    parser.add_argument("--faced-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--sequences-per-subject", type=int, default=4)
    args = parser.parse_args()
    datasets = {
        "ISRUC": dataset_statistics("ISRUC", args.isruc_root, args.seed, args.sequences_per_subject),
        "FACED": dataset_statistics("FACED", args.faced_root, args.seed, args.sequences_per_subject),
    }
    result = {"datasets": datasets, "cross_dataset": cross_dataset_rows(datasets["ISRUC"], datasets["FACED"])}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "INVARIANCE_ANALYSIS.json").write_text(json.dumps(result, indent=2))
    write_markdown(args.output_dir / "INVARIANCE_ANALYSIS.md", result)
    print(json.dumps({"components": len(COMPONENTS)}))


if __name__ == "__main__":
    main()
