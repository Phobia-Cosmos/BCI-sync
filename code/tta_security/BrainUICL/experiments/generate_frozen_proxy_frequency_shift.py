#!/usr/bin/env python3
"""Generate fixed bounded EEG uploads for the frequency/shift experiment.

The source-pretrained proxy never receives victim updates. It produces one
one-step-forgetting direction per selected sequence; four immutable upload
streams reuse those directions and vary only the attacked-task set and the
Rademacher sign distribution:

* I-NS: infrequent, non-shifted;
* I-S:  infrequent, shifted;
* F-NS: frequent, non-shifted;
* F-S:  frequent, shifted.

Unmodified arrays are symlinked to the canonical dataset. Modified arrays and
all experiment metadata remain under the experiment output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
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

from regularization_cl_attacks import brainwash_one_step_batch  # noqa: E402
from regularization_cl_eeg import build_split  # noqa: E402
from rttdp_brainuicl_full import (  # noqa: E402
    clone_blocks,
    flat_logits,
    forward_blocks,
    load_pretrained,
    merge_subject_paths,
    set_train,
    subject_paths,
)
from utils.config import ModelConfig  # noqa: E402
from utils.util import fix_randomness  # noqa: E402


STREAM_SPECS = {
    "I-NS": {"frequency": "infrequent", "shifted": False},
    "I-S": {"frequency": "infrequent", "shifted": True},
    "F-NS": {"frequency": "frequent", "shifted": False},
    "F-S": {"frequency": "frequent", "shifted": True},
}


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("At least one positive relative-L2 budget is required")
    return sorted(set(values))


def budget_name(relative_l2: float) -> str:
    scaled = int(round(relative_l2 * 10000))
    return f"rel_l2_{scaled:04d}"


def evenly_spaced_tasks(total_tasks: int, count: int) -> list[int]:
    if total_tasks < 1 or not 1 <= count <= total_tasks:
        raise ValueError("Task count must be in [1, total_tasks]")
    positions = np.rint(np.linspace(1, total_tasks, count)).astype(int).tolist()
    if len(set(positions)) != count:
        raise RuntimeError("Evenly spaced task construction produced duplicates")
    return positions


def nested_attack_tasks(
    total_tasks: int,
    frequent_count: int,
    infrequent_count: int,
) -> tuple[list[int], list[int]]:
    frequent = evenly_spaced_tasks(total_tasks, frequent_count)
    if not 1 <= infrequent_count <= frequent_count:
        raise ValueError("Infrequent count must be in [1, frequent_count]")
    if infrequent_count == frequent_count:
        return frequent, list(frequent)
    indices = np.rint(
        np.linspace(0, frequent_count - 1, infrequent_count + 2)[1:-1]
    ).astype(int)
    infrequent = [frequent[int(index)] for index in indices]
    if len(set(infrequent)) != infrequent_count:
        raise RuntimeError("Nested infrequent task construction produced duplicates")
    return frequent, infrequent


def balanced_rademacher(count: int, seed: int) -> np.ndarray:
    if count < 0:
        raise ValueError("Sign count cannot be negative")
    rng = np.random.default_rng(seed)
    values = np.asarray(
        [1] * (count // 2) + [-1] * (count // 2),
        dtype=np.int8,
    )
    if count % 2:
        values = np.concatenate(
            (values, np.asarray([rng.choice([-1, 1])], dtype=np.int8))
        )
    rng.shuffle(values)
    return values


def bandlimit_direction(
    direction: torch.Tensor,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
) -> torch.Tensor:
    if low_hz <= 0 and high_hz >= sample_rate / 2:
        return direction
    spectrum = torch.fft.rfft(direction, dim=-1)
    frequencies = torch.fft.rfftfreq(
        direction.shape[-1],
        d=1.0 / sample_rate,
        device=direction.device,
    )
    keep = (frequencies >= low_hz) & (frequencies <= high_hz)
    filtered = spectrum * keep.view(*([1] * (spectrum.ndim - 1)), -1)
    return torch.fft.irfft(filtered, n=direction.shape[-1], dim=-1)


def project_direction(
    direction: torch.Tensor,
    base: torch.Tensor,
    *,
    linf_std_scale: float,
    relative_l2: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Scale each sequence onto the intersection of L-inf and relative-L2 balls."""
    if direction.shape != base.shape or direction.ndim < 2:
        raise ValueError("Direction and base must have the same batched shape")
    flat_direction = direction.reshape(direction.shape[0], -1)
    flat_base = base.reshape(base.shape[0], -1)
    max_abs = flat_direction.abs().amax(dim=1)
    direction_norm = torch.linalg.vector_norm(flat_direction, dim=1)
    base_norm = torch.linalg.vector_norm(flat_base, dim=1)
    base_std = flat_base.std(dim=1, unbiased=False)
    linf_limit = base_std.clamp_min(eps) * linf_std_scale
    l2_limit = base_norm.clamp_min(eps) * relative_l2
    linf_scale = linf_limit / max_abs.clamp_min(eps)
    l2_scale = l2_limit / direction_norm.clamp_min(eps)
    scale = torch.minimum(linf_scale, l2_scale)
    scale = torch.where(direction_norm > eps, scale, torch.zeros_like(scale))
    return direction * scale.view(-1, *([1] * (direction.ndim - 1)))


def tensor_digest(blocks) -> str:
    digest = hashlib.sha256()
    for block in blocks:
        for name, tensor in sorted(block.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sequence_batch(paths: list[Path], device: torch.device):
    arrays = [torch.from_numpy(np.load(path).astype(np.float32)) for path in paths]
    batch = torch.stack(arrays).to(device)
    return batch[:, :, :2, :], batch[:, :, 2:, :]


def sequence_metrics(
    base: torch.Tensor,
    delta: torch.Tensor,
    *,
    sample_rate: float,
    band_low: float,
    band_high: float,
) -> dict[str, list[float]]:
    flat_base = base.reshape(base.shape[0], -1)
    flat_delta = delta.reshape(delta.shape[0], -1)
    base_norm = torch.linalg.vector_norm(flat_base, dim=1).clamp_min(1e-12)
    base_std = flat_base.std(dim=1, unbiased=False).clamp_min(1e-12)
    signal_spectrum = torch.fft.rfft(base, dim=-1)
    attacked_spectrum = torch.fft.rfft(base + delta, dim=-1)
    signal_power = signal_spectrum.abs().square().mean(dim=tuple(range(1, base.ndim - 1)))
    attacked_power = attacked_spectrum.abs().square().mean(
        dim=tuple(range(1, base.ndim - 1))
    )
    signal_distribution = signal_power / signal_power.sum(dim=1, keepdim=True).clamp_min(1e-12)
    attacked_distribution = attacked_power / attacked_power.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    spectral_total_variation = 0.5 * (
        signal_distribution - attacked_distribution
    ).abs().sum(dim=1)
    delta_power = torch.fft.rfft(delta, dim=-1).abs().square()
    frequencies = torch.fft.rfftfreq(
        delta.shape[-1],
        d=1.0 / sample_rate,
        device=delta.device,
    )
    outside = (frequencies < band_low) | (frequencies > band_high)
    outside_power = delta_power[..., outside].sum(dim=tuple(range(1, delta_power.ndim)))
    total_delta_power = delta_power.sum(dim=tuple(range(1, delta_power.ndim))).clamp_min(1e-12)
    base_min = base.amin(dim=tuple(range(1, base.ndim)), keepdim=True)
    base_max = base.amax(dim=tuple(range(1, base.ndim)), keepdim=True)
    attacked = base + delta
    outside_range = (attacked < base_min) | (attacked > base_max)
    return {
        "relative_l2": (
            torch.linalg.vector_norm(flat_delta, dim=1) / base_norm
        ).detach().cpu().tolist(),
        "linf_over_std": (
            flat_delta.abs().amax(dim=1) / base_std
        ).detach().cpu().tolist(),
        "spectral_total_variation": spectral_total_variation.detach().cpu().tolist(),
        "perturbation_out_of_band_fraction": (
            outside_power / total_delta_power
        ).detach().cpu().tolist(),
        "sample_outside_clean_range_fraction": outside_range.float()
        .reshape(base.shape[0], -1)
        .mean(dim=1)
        .detach()
        .cpu()
        .tolist(),
    }


def aggregate(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "max": float(array.max())}


def replace_with_symlink(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def save_array(destination: Path, array: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    np.save(destination, array.astype(np.float32, copy=False))


def direction_cache_paths(root: Path, task_index: int, indices: list[int]) -> list[Path]:
    task_root = root / f"individual_{task_index}"
    return [task_root / f"{index}.npy" for index in indices]


def generate_task_directions(
    *,
    args,
    proxy_blocks,
    label_blocks,
    current_paths: list[Path],
    reference_paths: list[Path],
    selected_indices: list[int],
    task_index: int,
    subject: int,
    cache_root: Path,
) -> tuple[dict[int, torch.Tensor], list[dict[str, float]]]:
    cache_paths = direction_cache_paths(cache_root, task_index, selected_indices)
    if cache_paths and all(path.is_file() for path in cache_paths):
        cached = {
            index: torch.from_numpy(np.load(path).astype(np.float32)).to(args.device)
            for index, path in zip(selected_indices, cache_paths)
        }
        return cached, []

    directions: dict[int, torch.Tensor] = {}
    diagnostics: list[dict[str, float]] = []
    for batch_start in range(0, len(selected_indices), args.direction_batch):
        batch_indices = selected_indices[
            batch_start : batch_start + args.direction_batch
        ]
        eog, eeg = load_sequence_batch(
            [current_paths[index] for index in batch_indices],
            args.device,
        )
        reference_start = (
            args.attack_seed
            + 37 * task_index
            + 101 * batch_start
            + 1009 * subject
        ) % len(reference_paths)
        reference_indices = [
            (reference_start + offset) % len(reference_paths)
            for offset in range(args.reference_batch)
        ]
        reference_eog, reference_eeg = load_sequence_batch(
            [reference_paths[index] for index in reference_indices],
            args.device,
        )
        eog_adv, eeg_adv, row = brainwash_one_step_batch(
            proxy_blocks,
            label_blocks,
            eog,
            eeg,
            reference_eog,
            reference_eeg,
            args,
        )
        delta_eog = bandlimit_direction(
            eog_adv - eog,
            args.sample_rate,
            args.band_low,
            args.band_high,
        )
        delta_eeg = bandlimit_direction(
            eeg_adv - eeg,
            args.sample_rate,
            args.band_low,
            args.band_high,
        )
        for local_index, sequence_index in enumerate(batch_indices):
            combined = torch.cat(
                (delta_eog[local_index], delta_eeg[local_index]),
                dim=1,
            ).detach()
            if float(torch.linalg.vector_norm(combined).cpu()) <= 1e-12:
                raise RuntimeError(
                    f"Proxy produced a zero direction for task {task_index}, "
                    f"sequence {sequence_index}"
                )
            cache_path = cache_root / f"individual_{task_index}" / f"{sequence_index}.npy"
            save_array(cache_path, combined.cpu().numpy())
            directions[sequence_index] = combined.to(args.device)
        diagnostics.append({key: float(value) for key, value in row.items()})
    return directions, diagnostics


@torch.no_grad()
def proxy_labels(proxy_blocks, eog: torch.Tensor, eeg: torch.Tensor, args) -> torch.Tensor:
    set_train(proxy_blocks, False)
    return flat_logits(forward_blocks(proxy_blocks, eog, eeg, args)).argmax(dim=1)


def materialize_task_stream(
    *,
    args,
    proxy_blocks,
    stream_name: str,
    stream_root: Path,
    budget: float,
    task_index: int,
    subject: int,
    current_paths: list[Path],
    selected_indices: list[int],
    directions: dict[int, torch.Tensor],
    attack_tasks: set[int],
    sign_map: dict[int, int],
) -> dict:
    task_root = stream_root / f"individual_{task_index}"
    data_root = task_root / "data"
    attacked = task_index in attack_tasks
    active_indices = selected_indices if attacked else []
    active_set = set(active_indices)

    for index, source in enumerate(current_paths):
        if index not in active_set:
            replace_with_symlink(data_root / f"{index}.npy", source)

    rel_eog_values: list[float] = []
    rel_eeg_values: list[float] = []
    linf_eog_values: list[float] = []
    linf_eeg_values: list[float] = []
    spectral_eog_values: list[float] = []
    spectral_eeg_values: list[float] = []
    out_of_band_eog_values: list[float] = []
    out_of_band_eeg_values: list[float] = []
    outside_range_eog_values: list[float] = []
    outside_range_eeg_values: list[float] = []
    clean_labels: list[torch.Tensor] = []
    attacked_labels: list[torch.Tensor] = []
    delta_sum = None
    delta_energy = 0.0

    for batch_start in range(0, len(active_indices), args.materialize_batch):
        batch_indices = active_indices[
            batch_start : batch_start + args.materialize_batch
        ]
        eog, eeg = load_sequence_batch(
            [current_paths[index] for index in batch_indices],
            args.device,
        )
        cached = torch.stack([directions[index] for index in batch_indices])
        raw_eog = cached[:, :, :2, :]
        raw_eeg = cached[:, :, 2:, :]
        delta_eog = project_direction(
            raw_eog,
            eog,
            linf_std_scale=args.linf_std_scale,
            relative_l2=budget,
        )
        delta_eeg = project_direction(
            raw_eeg,
            eeg,
            linf_std_scale=args.linf_std_scale,
            relative_l2=budget,
        )
        signs = torch.tensor(
            [sign_map[index] for index in batch_indices],
            device=args.device,
            dtype=eog.dtype,
        ).view(-1, 1, 1, 1)
        delta_eog = delta_eog * signs
        delta_eeg = delta_eeg * signs
        eog_adv = eog + delta_eog
        eeg_adv = eeg + delta_eeg

        clean_labels.append(proxy_labels(proxy_blocks, eog, eeg, args).cpu())
        attacked_labels.append(proxy_labels(proxy_blocks, eog_adv, eeg_adv, args).cpu())
        metrics_eog = sequence_metrics(
            eog,
            delta_eog,
            sample_rate=args.sample_rate,
            band_low=args.band_low,
            band_high=args.band_high,
        )
        metrics_eeg = sequence_metrics(
            eeg,
            delta_eeg,
            sample_rate=args.sample_rate,
            band_low=args.band_low,
            band_high=args.band_high,
        )
        rel_eog_values.extend(metrics_eog["relative_l2"])
        rel_eeg_values.extend(metrics_eeg["relative_l2"])
        linf_eog_values.extend(metrics_eog["linf_over_std"])
        linf_eeg_values.extend(metrics_eeg["linf_over_std"])
        spectral_eog_values.extend(metrics_eog["spectral_total_variation"])
        spectral_eeg_values.extend(metrics_eeg["spectral_total_variation"])
        out_of_band_eog_values.extend(
            metrics_eog["perturbation_out_of_band_fraction"]
        )
        out_of_band_eeg_values.extend(
            metrics_eeg["perturbation_out_of_band_fraction"]
        )
        outside_range_eog_values.extend(
            metrics_eog["sample_outside_clean_range_fraction"]
        )
        outside_range_eeg_values.extend(
            metrics_eeg["sample_outside_clean_range_fraction"]
        )

        combined_delta = torch.cat((delta_eog, delta_eeg), dim=2)
        batch_sum = combined_delta.sum(dim=0).double()
        delta_sum = batch_sum if delta_sum is None else delta_sum + batch_sum
        delta_energy += float(combined_delta.double().pow(2).sum().cpu())
        for local_index, sequence_index in enumerate(batch_indices):
            upload = torch.cat(
                (eog_adv[local_index], eeg_adv[local_index]),
                dim=1,
            )
            save_array(
                data_root / f"{sequence_index}.npy",
                upload.detach().cpu().numpy(),
            )

    if clean_labels:
        clean_flat = torch.cat(clean_labels)
        attacked_flat = torch.cat(attacked_labels)
        preservation = float((clean_flat == attacked_flat).float().mean())
    else:
        preservation = 1.0
    if delta_sum is None or delta_energy <= 0:
        empirical_mean_ratio = 0.0
    else:
        mean_delta = delta_sum / len(active_indices)
        rms_delta = math.sqrt(delta_energy / len(active_indices))
        empirical_mean_ratio = float(
            torch.linalg.vector_norm(mean_delta).cpu() / max(rms_delta, 1e-12)
        )

    metadata = {
        "task": int(task_index),
        "subject": int(subject),
        "stream": stream_name,
        "frequency": STREAM_SPECS[stream_name]["frequency"],
        "shifted": bool(STREAM_SPECS[stream_name]["shifted"]),
        "uploaded": int(len(current_paths)),
        "noisy_sequences": int(len(active_indices)),
        "noise_fraction": float(len(active_indices) / max(len(current_paths), 1)),
        "noise_indices": [int(index) for index in active_indices],
        "signs": {str(index): int(sign_map[index]) for index in active_indices},
        "sign_positive": int(sum(sign_map[index] > 0 for index in active_indices)),
        "sign_negative": int(sum(sign_map[index] < 0 for index in active_indices)),
        "relative_l2_budget": float(budget),
        "linf_std_scale": float(args.linf_std_scale),
        "eog_relative_l2": aggregate(rel_eog_values),
        "eeg_relative_l2": aggregate(rel_eeg_values),
        "eog_linf_over_std": aggregate(linf_eog_values),
        "eeg_linf_over_std": aggregate(linf_eeg_values),
        "eog_spectral_total_variation": aggregate(spectral_eog_values),
        "eeg_spectral_total_variation": aggregate(spectral_eeg_values),
        "eog_perturbation_out_of_band_fraction": aggregate(out_of_band_eog_values),
        "eeg_perturbation_out_of_band_fraction": aggregate(out_of_band_eeg_values),
        "eog_sample_outside_clean_range_fraction": aggregate(
            outside_range_eog_values
        ),
        "eeg_sample_outside_clean_range_fraction": aggregate(
            outside_range_eeg_values
        ),
        "proxy_pseudo_label_preservation": preservation,
        "empirical_sequence_mean_ratio": empirical_mean_ratio,
        "clean_source": str(args.data_root / str(subject) / "data"),
    }
    (task_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    return metadata


def make_attack_args(parsed) -> SimpleNamespace:
    values = vars(parsed).copy()
    values.update(
        {
            "attack_mode": "brainwash_reckless",
            "attack_eps_scale": parsed.direction_eps_std_scale,
            "attack_steps": parsed.direction_steps,
            "attack_inner_lr": parsed.direction_inner_lr,
            "attack_cautious_weight": 0.0,
            "attack_param_scope": "classifier",
            "attack_random_start": False,
        }
    )
    return SimpleNamespace(**values)


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
        default=REPO_ROOT / "experiments" / "frozen_proxy_frequency_shift" / "seed4321",
    )
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--attack-seed", type=int, default=1701)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--frequent-count", type=int, default=25)
    parser.add_argument("--infrequent-count", type=int, default=3)
    parser.add_argument("--sequence-fraction", type=float, default=0.20)
    parser.add_argument("--relative-l2-budgets", type=str, default="0.05")
    parser.add_argument("--linf-std-scale", type=float, default=0.10)
    parser.add_argument("--direction-eps-std-scale", type=float, default=0.20)
    parser.add_argument("--direction-steps", type=int, default=5)
    parser.add_argument("--direction-inner-lr", type=float, default=1e-4)
    parser.add_argument("--direction-batch", type=int, default=3)
    parser.add_argument("--materialize-batch", type=int, default=12)
    parser.add_argument("--reference-batch", type=int, default=1)
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--band-low", type=float, default=0.3)
    parser.add_argument("--band-high", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    parsed = parse_args()
    if not 0 < parsed.sequence_fraction <= 1:
        raise ValueError("--sequence-fraction must be in (0, 1]")
    if parsed.linf_std_scale <= 0 or parsed.direction_eps_std_scale <= 0:
        raise ValueError("L-inf scales must be positive")
    if parsed.direction_steps < 1 or parsed.direction_batch < 1:
        raise ValueError("Direction steps and batch must be positive")
    if parsed.materialize_batch < 1 or parsed.reference_batch < 1:
        raise ValueError("Materialization and reference batches must be positive")
    if not 0 <= parsed.band_low < parsed.band_high <= parsed.sample_rate / 2:
        raise ValueError("Band limits must satisfy 0 <= low < high <= Nyquist")
    budgets = parse_float_list(parsed.relative_l2_budgets)
    if parsed.overwrite and parsed.output_root.exists():
        shutil.rmtree(parsed.output_root)
    parsed.output_root.mkdir(parents=True, exist_ok=True)

    args = make_attack_args(parsed)
    args.dataset = "ISRUC"
    args.model_param = ModelConfig(args.dataset)
    args.device = torch.device(
        f"cuda:{args.gpu}"
        if args.gpu >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    fix_randomness(args.seed)
    split = build_split(args)
    total_tasks = len(split["new_order"])
    frequent_tasks, infrequent_tasks = nested_attack_tasks(
        total_tasks,
        min(parsed.frequent_count, total_tasks),
        min(parsed.infrequent_count, parsed.frequent_count, total_tasks),
    )
    frequent_set = set(frequent_tasks)
    infrequent_set = set(infrequent_tasks)

    proxy_blocks = load_pretrained(args)
    label_blocks = clone_blocks(proxy_blocks, args)
    set_train(proxy_blocks, False)
    set_train(label_blocks, False)
    initial_digest = tensor_digest(proxy_blocks)
    reference_paths = merge_subject_paths(args.data_root, split["train_idx"])[0]
    if not reference_paths:
        raise RuntimeError("No source reference inputs were found")

    checkpoint_dir = args.input_checkpoint_root / args.dataset / "Pretrain"
    checkpoint_hashes = {
        path.name: sha256_file(path)
        for path in sorted(checkpoint_dir.glob(f"*_{args.seed}.pkl"))
    }
    config = {
        "seed": args.seed,
        "attack_seed": args.attack_seed,
        "data_root": str(args.data_root),
        "input_checkpoint_root": str(args.input_checkpoint_root),
        "output_root": str(args.output_root),
        "proxy": "source-pretrained frozen EEG model",
        "direction_objective": "one-step source-proxy retention loss maximization",
        "total_tasks": total_tasks,
        "frequent_tasks": frequent_tasks,
        "infrequent_tasks": infrequent_tasks,
        "sequence_fraction": args.sequence_fraction,
        "relative_l2_budgets": budgets,
        "linf_std_scale": args.linf_std_scale,
        "direction_eps_std_scale": args.direction_eps_std_scale,
        "direction_steps": args.direction_steps,
        "direction_inner_lr": args.direction_inner_lr,
        "sample_rate": args.sample_rate,
        "band_hz": [args.band_low, args.band_high],
        "split": split,
        "checkpoint_hashes": checkpoint_hashes,
    }
    config_path = args.output_root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(
            f"Existing configuration differs at {config_path}; use --overwrite"
        )
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    direction_root = args.output_root / "directions"
    stream_summaries = {
        budget_name(budget): {
            stream: {"tasks": 0, "noisy_sequences": 0, "uploaded": 0}
            for stream in STREAM_SPECS
        }
        for budget in budgets
    }
    direction_rows: list[dict] = []

    for task_index, subject in enumerate(split["new_order"], start=1):
        current_paths = subject_paths(args.data_root, subject)[0]
        poison_count = min(
            len(current_paths),
            int(math.ceil(len(current_paths) * args.sequence_fraction)),
        )
        sequence_rng = np.random.default_rng(
            args.attack_seed + 1009 * task_index + 9173 * int(subject)
        )
        selected_indices = sorted(
            sequence_rng.choice(
                len(current_paths), poison_count, replace=False
            ).astype(int).tolist()
        )
        sign_values = balanced_rademacher(
            len(selected_indices),
            args.attack_seed + 7919 * task_index + int(subject),
        )
        nonshifted_signs = {
            index: int(sign)
            for index, sign in zip(selected_indices, sign_values)
        }
        shifted_signs = {index: 1 for index in selected_indices}

        if task_index in frequent_set:
            directions, proxy_rows = generate_task_directions(
                args=args,
                proxy_blocks=proxy_blocks,
                label_blocks=label_blocks,
                current_paths=current_paths,
                reference_paths=reference_paths,
                selected_indices=selected_indices,
                task_index=task_index,
                subject=subject,
                cache_root=direction_root,
            )
            direction_rows.extend(proxy_rows)
        else:
            directions = {}

        for budget in budgets:
            budget_key = budget_name(budget)
            for stream_name, spec in STREAM_SPECS.items():
                attack_tasks = (
                    infrequent_set
                    if spec["frequency"] == "infrequent"
                    else frequent_set
                )
                metadata = materialize_task_stream(
                    args=args,
                    proxy_blocks=proxy_blocks,
                    stream_name=stream_name,
                    stream_root=args.output_root / budget_key / stream_name,
                    budget=budget,
                    task_index=task_index,
                    subject=subject,
                    current_paths=current_paths,
                    selected_indices=selected_indices,
                    directions=directions,
                    attack_tasks=attack_tasks,
                    sign_map=(shifted_signs if spec["shifted"] else nonshifted_signs),
                )
                summary = stream_summaries[budget_key][stream_name]
                summary["tasks"] += int(metadata["noisy_sequences"] > 0)
                summary["noisy_sequences"] += metadata["noisy_sequences"]
                summary["uploaded"] += metadata["uploaded"]

        print(
            f"[frozen-proxy] task={task_index}/{total_tasks} subject={subject} "
            f"directions={len(directions)}",
            flush=True,
        )

    final_digest = tensor_digest(proxy_blocks)
    if final_digest != initial_digest:
        raise RuntimeError("Frozen proxy parameters changed during stream generation")
    diagnostics_mean = {}
    if direction_rows:
        for key in direction_rows[0]:
            diagnostics_mean[key] = float(
                np.mean([row[key] for row in direction_rows])
            )
    manifest = {
        "config": config,
        "proxy_parameter_sha256_before": initial_digest,
        "proxy_parameter_sha256_after": final_digest,
        "proxy_parameters_unchanged": True,
        "stream_summaries": stream_summaries,
        "direction_diagnostics_mean": diagnostics_mean,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
