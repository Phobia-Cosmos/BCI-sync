#!/usr/bin/env python3
"""Visualize the processed ISRUC and FACED inputs used by the EEG CL runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram, welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATASETS = {
    "ISRUC": {
        "default_root": Path(
            "/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32"
        ),
        "sfreq": 100,
        "epoch_seconds": 30,
        "class_names": ["W", "N1", "N2", "N3", "REM"],
        "channels": ["EOG-L", "EOG-R", "F3", "C3", "O1", "F4", "C4", "O2"],
        "eeg_start": 2,
        "display_channels": [0, 1, 2, 3, 4, 5, 6, 7],
        "spectral_channel": 3,
        "max_frequency": 45,
        "unit_scale": 1e6,
        "unit": "uV",
    },
    "FACED": {
        "default_root": Path("/home/undefined/Disk/datasets/FACED_processed"),
        "sfreq": 250,
        "epoch_seconds": 10,
        "class_names": [
            "Anger",
            "Disgust",
            "Fear",
            "Sadness",
            "Neutral",
            "Amusement",
            "Inspiration",
            "Joy",
            "Tenderness",
        ],
        "channels": [
            "Fp1",
            "Fp2",
            "Fz",
            "F3",
            "F4",
            "F7",
            "F8",
            "FC1",
            "FC2",
            "FC5",
            "FC6",
            "Cz",
            "C3",
            "C4",
            "T3/T7",
            "T4/T8",
            "A1/CP1",
            "A2/CP2",
            "CP1/CP5",
            "CP2/CP6",
            "CP5/Pz",
            "CP6/P3",
            "Pz/P4",
            "P3/P7",
            "P4/P8",
            "T5/PO3",
            "T6/PO4",
            "PO3/Oz",
            "PO4/O1",
            "Oz/O2",
            "O1/HEOR",
            "O2/HEOL",
        ],
        "eeg_start": 0,
        "display_channels": [0, 3, 12, 23, 30, 4, 13, 31],
        "spectral_channel": 12,
        "max_frequency": 45,
        "unit_scale": 1.0,
        "unit": "epoch z-score",
    },
}


def numeric_key(path: Path) -> tuple[int, str]:
    digits = "".join(char for char in path.name if char.isdigit())
    return (int(digits) if digits else -1, path.name)


def discover_subjects(root: Path) -> list[Path]:
    subjects = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "data").is_dir() and (path / "label").is_dir()
    ]
    return sorted(subjects, key=numeric_key)


def sequence_paths(subject: Path) -> list[Path]:
    return sorted((subject / "data").glob("*.npy"), key=numeric_key)


def load_sequence(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    labels = np.load(path.parent.parent / "label" / path.name, allow_pickle=False).astype(
        np.int64, copy=False
    )
    return values, labels


def epoch_psd(epoch: np.ndarray, sfreq: int, eeg_start: int) -> tuple[np.ndarray, np.ndarray]:
    eeg = epoch[eeg_start:]
    nperseg = min(eeg.shape[-1], sfreq * 4)
    frequencies, power = welch(eeg, fs=sfreq, nperseg=nperseg, axis=-1)
    return frequencies, power.mean(axis=0)


def spectral_descriptor(sequence: np.ndarray, config: dict) -> np.ndarray:
    sfreq = config["sfreq"]
    eeg = sequence[:4, config["eeg_start"] :]
    nperseg = min(eeg.shape[-1], sfreq * 2)
    frequencies, power = welch(eeg, fs=sfreq, nperseg=nperseg, axis=-1)
    power = power.mean(axis=(0, 1))
    valid = (frequencies >= 0.5) & (frequencies <= config["max_frequency"])
    total = np.trapezoid(power[valid], frequencies[valid]) + 1e-12
    bands = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45))
    relative = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        relative.append(np.trapezoid(power[mask], frequencies[mask]) / total)
    normalized = power[valid] / (power[valid].sum() + 1e-12)
    entropy = -np.sum(normalized * np.log(normalized + 1e-12))
    return np.asarray(relative + [entropy], dtype=np.float64)


def collect_metadata(subjects: list[Path], classes: int) -> tuple[np.ndarray, np.ndarray]:
    class_counts = np.zeros(classes, dtype=np.int64)
    sequence_counts = []
    for subject in subjects:
        paths = sequence_paths(subject)
        sequence_counts.append(len(paths))
        for path in paths:
            labels = np.load(
                path.parent.parent / "label" / path.name, allow_pickle=False
            ).astype(np.int64, copy=False)
            class_counts += np.bincount(labels, minlength=classes)
    return class_counts, np.asarray(sequence_counts, dtype=np.int64)


def collect_class_psd(
    subjects: list[Path], config: dict, max_epochs_per_class: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    buckets: list[list[np.ndarray]] = [[] for _ in config["class_names"]]
    frequencies = None
    for subject in subjects:
        for path in sequence_paths(subject):
            values, labels = load_sequence(path)
            for epoch, label in zip(values, labels):
                label = int(label)
                if len(buckets[label]) >= max_epochs_per_class:
                    continue
                frequencies, power = epoch_psd(
                    epoch, config["sfreq"], config["eeg_start"]
                )
                buckets[label].append(power)
            if all(len(bucket) >= max_epochs_per_class for bucket in buckets):
                break
        if all(len(bucket) >= max_epochs_per_class for bucket in buckets):
            break
    if frequencies is None:
        raise RuntimeError("No epochs found for PSD visualization")
    return frequencies, [np.mean(bucket, axis=0) for bucket in buckets]


def plot_sample_overview(
    dataset: str,
    subject: Path,
    path: Path,
    config: dict,
    output: Path,
    psd_subjects: list[Path],
) -> dict:
    values, labels = load_sequence(path)
    sfreq = config["sfreq"]
    display_seconds = min(8, config["epoch_seconds"])
    display_samples = sfreq * display_seconds
    channel_indices = config["display_channels"]
    scaled = values[0, channel_indices, :display_samples] * config["unit_scale"]

    figure = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[2.25, 1.3, 1.35])
    waveform_grid = grid[0, :].subgridspec(len(channel_indices), 1, hspace=0.05)
    time = np.arange(display_samples) / sfreq
    for row, channel_index in enumerate(channel_indices):
        axis = figure.add_subplot(waveform_grid[row, 0])
        axis.plot(time, scaled[row], color="#176B87", linewidth=0.65)
        axis.set_ylabel(config["channels"][channel_index], rotation=0, ha="right", va="center")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", left=False, labelleft=False)
        if row != len(channel_indices) - 1:
            axis.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            axis.set_xlabel("Time (s)")
        if row == 0:
            axis.set_title(
                f"{dataset} processed input: {subject.name}, sequence {path.stem}, "
                f"epoch 0 ({config['unit']})",
                loc="left",
                fontweight="bold",
            )

    spectral_channel = config["spectral_channel"]
    frequency, segment_time, spectrum = spectrogram(
        values[0, spectral_channel],
        fs=sfreq,
        nperseg=min(values.shape[-1], sfreq * 2),
        noverlap=min(values.shape[-1], int(sfreq * 1.5)),
        scaling="density",
        mode="psd",
    )
    mask = (frequency >= 0.5) & (frequency <= config["max_frequency"])
    axis = figure.add_subplot(grid[1, 0])
    image = axis.pcolormesh(
        segment_time,
        frequency[mask],
        10 * np.log10(spectrum[mask] + 1e-20),
        shading="auto",
        cmap="magma",
    )
    axis.set_title(f"Spectrogram: {config['channels'][spectral_channel]}", loc="left")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    figure.colorbar(image, ax=axis, label="PSD (dB)")

    axis = figure.add_subplot(grid[1, 1])
    frequencies, class_power = collect_class_psd(
        subjects=psd_subjects, config=config, max_epochs_per_class=12
    )
    valid = (frequencies >= 0.5) & (frequencies <= config["max_frequency"])
    for name, power in zip(config["class_names"], class_power):
        axis.plot(frequencies[valid], 10 * np.log10(power[valid] + 1e-20), label=name)
    axis.set_title("Mean PSD by label (sampled dataset epochs)", loc="left")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (dB)")
    axis.legend(ncol=3, fontsize=8, frameon=False)

    frequency_rows = []
    row_frequencies = None
    for epoch in values:
        row_frequencies, row_power = epoch_psd(epoch, sfreq, config["eeg_start"])
        frequency_rows.append(10 * np.log10(row_power + 1e-20))
    frequency_rows = np.stack(frequency_rows)
    valid = (row_frequencies >= 0.5) & (row_frequencies <= config["max_frequency"])
    axis = figure.add_subplot(grid[2, 0])
    image = axis.imshow(
        frequency_rows[:, valid],
        aspect="auto",
        origin="lower",
        extent=[row_frequencies[valid][0], row_frequencies[valid][-1], -0.5, len(labels) - 0.5],
        cmap="viridis",
    )
    axis.set_title("Sequence view: one PSD row per epoch", loc="left")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Epoch index")
    figure.colorbar(image, ax=axis, label="PSD (dB)")

    axis = figure.add_subplot(grid[2, 1])
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(config["class_names"])))
    for index, label in enumerate(labels):
        axis.barh(0, 1, left=index, color=colors[int(label)], edgecolor="white", linewidth=0.5)
        axis.text(index + 0.5, 0, str(int(label)), ha="center", va="center", fontsize=8)
    axis.set_xlim(0, len(labels))
    axis.set_ylim(-0.8, 0.8)
    axis.set_yticks([])
    axis.set_xticks(np.arange(0, len(labels) + 1, 2))
    axis.set_xlabel("Epoch index")
    axis.set_title("Ground-truth label timeline", loc="left")
    handles = [
        plt.Line2D([0], [0], color=colors[index], linewidth=7, label=f"{index}: {name}")
        for index, name in enumerate(config["class_names"])
    ]
    axis.legend(handles=handles, ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))

    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return {
        "subject": subject.name,
        "sequence": int(path.stem),
        "shape": list(values.shape),
        "labels": labels.tolist(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def plot_dataset_summary(
    dataset: str,
    subjects: list[Path],
    config: dict,
    output: Path,
    descriptors_per_subject: int,
) -> dict:
    class_counts, sequence_counts = collect_metadata(subjects, len(config["class_names"]))
    descriptors = []
    majority_labels = []
    subject_ids = []
    for subject_index, subject in enumerate(subjects):
        paths = sequence_paths(subject)
        if len(paths) > descriptors_per_subject:
            selected_indices = np.linspace(0, len(paths) - 1, descriptors_per_subject, dtype=int)
            paths = [paths[index] for index in selected_indices]
        for path in paths:
            values, labels = load_sequence(path)
            descriptors.append(spectral_descriptor(values, config))
            majority_labels.append(int(np.bincount(labels).argmax()))
            subject_ids.append(subject_index)
    descriptors = StandardScaler().fit_transform(np.stack(descriptors))
    coordinates = PCA(n_components=2, random_state=0).fit_transform(descriptors)

    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(config["class_names"])))
    axes[0, 0].bar(config["class_names"], class_counts, color=colors)
    axes[0, 0].set_title(f"{dataset}: epoch label distribution", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Epochs")
    axes[0, 0].tick_params(axis="x", rotation=35)

    axes[0, 1].hist(sequence_counts, bins=np.arange(sequence_counts.min(), sequence_counts.max() + 2) - 0.5, color="#2A9D8F")
    axes[0, 1].axvline(np.median(sequence_counts), color="#D1495B", linestyle="--", label=f"median={np.median(sequence_counts):.0f}")
    axes[0, 1].set_title("Sequences per subject", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Sequence count")
    axes[0, 1].set_ylabel("Subjects")
    axes[0, 1].legend(frameon=False)

    for label, name in enumerate(config["class_names"]):
        mask = np.asarray(majority_labels) == label
        axes[1, 0].scatter(coordinates[mask, 0], coordinates[mask, 1], s=18, alpha=0.7, label=name, color=colors[label])
    axes[1, 0].set_title("Sequence spectral descriptors: colored by majority label", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("PCA 1")
    axes[1, 0].set_ylabel("PCA 2")
    axes[1, 0].legend(ncol=3, fontsize=8, frameon=False)

    scatter = axes[1, 1].scatter(coordinates[:, 0], coordinates[:, 1], c=subject_ids, cmap="turbo", s=18, alpha=0.75)
    axes[1, 1].set_title("Same descriptors: colored by subject", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("PCA 1")
    axes[1, 1].set_ylabel("PCA 2")
    figure.colorbar(scatter, ax=axes[1, 1], label="Subject index")

    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return {
        "subjects": len(subjects),
        "sequences": int(sequence_counts.sum()),
        "epochs": int(sequence_counts.sum() * 20),
        "sequences_per_subject": {
            "minimum": int(sequence_counts.min()),
            "median": float(np.median(sequence_counts)),
            "maximum": int(sequence_counts.max()),
        },
        "class_counts": {
            name: int(count) for name, count in zip(config["class_names"], class_counts)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isruc-root", type=Path, default=DATASETS["ISRUC"]["default_root"])
    parser.add_argument("--faced-root", type=Path, default=DATASETS["FACED"]["default_root"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/eeg_dataset_visualization"),
    )
    parser.add_argument("--descriptors-per-subject", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = {"ISRUC": args.isruc_root, "FACED": args.faced_root}
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for dataset, root in roots.items():
        config = DATASETS[dataset]
        subjects = discover_subjects(root)
        if not subjects:
            raise RuntimeError(f"No processed subjects found under {root}")
        subject = subjects[0]
        path = sequence_paths(subject)[0]
        sample = plot_sample_overview(
            dataset,
            subject,
            path,
            config,
            args.output_root / f"{dataset.lower()}_sample_overview.png",
            subjects,
        )
        statistics = plot_dataset_summary(
            dataset,
            subjects,
            config,
            args.output_root / f"{dataset.lower()}_dataset_summary.png",
            args.descriptors_per_subject,
        )
        summary[dataset] = {
            "root": str(root),
            "sampling_rate": config["sfreq"],
            "epoch_seconds": config["epoch_seconds"],
            "sequence_length": 20,
            "channels": len(config["channels"]),
            "display_unit": config["unit"],
            "sample": sample,
            "statistics": statistics,
        }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
