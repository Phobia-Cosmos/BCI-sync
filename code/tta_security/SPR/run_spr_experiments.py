#!/usr/bin/env python3
"""Run SPR experiment matrices reproducibly.

The official SPR entrypoint appends $SLURM_JOB_ID to --log-dir.  This wrapper
sets that id explicitly so local runs get stable names and can be skipped when
already completed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/undefined/Disk/ai-storage/.venv-rttdp-py39/bin/python")
DEFAULT_CHECKPOINT_ROOT = Path("/home/undefined/Disk/ai-storage/SPR_checkpoints")


@dataclass(frozen=True)
class Experiment:
    dataset: str
    noise: str
    corruption: str | None
    config: str
    episode: str
    overrides: tuple[str, ...]

    @property
    def run_key(self) -> str:
        suffix = "real" if self.corruption is None else f"{int(float(self.corruption) * 100)}"
        return f"{self.dataset}_{self.noise}{suffix}"


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("mnist", "sym", "0.2", "configs/mnist_spr.yaml", "episodes/mnist-split_epc1_a.yaml", ("asymmetric_noise=False", "corruption_percent=0.2")),
    Experiment("mnist", "sym", "0.4", "configs/mnist_spr.yaml", "episodes/mnist-split_epc1_a.yaml", ("asymmetric_noise=False", "corruption_percent=0.4")),
    Experiment("mnist", "sym", "0.6", "configs/mnist_spr.yaml", "episodes/mnist-split_epc1_a.yaml", ("asymmetric_noise=False", "corruption_percent=0.6")),
    Experiment("mnist", "asym", "0.2", "configs/mnist_spr.yaml", "episodes/mnist-split_epc1_asym_a.yaml", ("asymmetric_noise=True", "corruption_percent=0.2")),
    Experiment("mnist", "asym", "0.4", "configs/mnist_spr.yaml", "episodes/mnist-split_epc1_asym_a.yaml", ("asymmetric_noise=True", "corruption_percent=0.4")),
    Experiment("cifar10", "sym", "0.2", "configs/cifar10_spr.yaml", "episodes/cifar10-split_epc1_a.yaml", ("asymmetric_noise=False", "corruption_percent=0.2")),
    Experiment("cifar10", "sym", "0.4", "configs/cifar10_spr.yaml", "episodes/cifar10-split_epc1_a.yaml", ("asymmetric_noise=False", "corruption_percent=0.4")),
    Experiment("cifar10", "sym", "0.6", "configs/cifar10_spr.yaml", "episodes/cifar10-split_epc1_a.yaml", ("asymmetric_noise=False", "corruption_percent=0.6")),
    Experiment("cifar10", "asym", "0.2", "configs/cifar10_spr.yaml", "episodes/cifar10-split_epc1_asym_a.yaml", ("asymmetric_noise=True", "corruption_percent=0.2")),
    Experiment("cifar10", "asym", "0.4", "configs/cifar10_spr.yaml", "episodes/cifar10-split_epc1_asym_a.yaml", ("asymmetric_noise=True", "corruption_percent=0.4")),
    Experiment("cifar100", "rndsym", "0.2", "configs/cifar100_spr.yaml", "episodes/cifar100rnd-split_epc1_a.yaml", ("superclass_noise=False", "corruption_percent=0.2")),
    Experiment("cifar100", "rndsym", "0.4", "configs/cifar100_spr.yaml", "episodes/cifar100rnd-split_epc1_a.yaml", ("superclass_noise=False", "corruption_percent=0.4")),
    Experiment("cifar100", "rndsym", "0.6", "configs/cifar100_spr.yaml", "episodes/cifar100rnd-split_epc1_a.yaml", ("superclass_noise=False", "corruption_percent=0.6")),
    Experiment("cifar100", "supsym", "0.2", "configs/cifar100_spr.yaml", "episodes/cifar100sup-split_epc1_a.yaml", ("superclass_noise=True", "corruption_percent=0.2")),
    Experiment("cifar100", "supsym", "0.4", "configs/cifar100_spr.yaml", "episodes/cifar100sup-split_epc1_a.yaml", ("superclass_noise=True", "corruption_percent=0.4")),
    Experiment("cifar100", "supsym", "0.6", "configs/cifar100_spr.yaml", "episodes/cifar100sup-split_epc1_a.yaml", ("superclass_noise=True", "corruption_percent=0.6")),
    Experiment("webvision", "real", None, "configs/webvision_spr.yaml", "episodes/webvision-split_epc1_a.yaml", ()),
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_dir_for(log_root: Path, run_id: str) -> Path:
    return log_root / run_id


def has_result(run_dir: Path) -> bool:
    return any(run_dir.glob("events.out.tfevents*")) and (run_dir / "ckpts").exists()


def build_override(exp: Experiment, args: argparse.Namespace) -> str:
    pieces = list(exp.overrides)
    if args.suite in {"quick", "medium", "large"}:
        pieces.extend(
            [
                f"expert_train_epochs={args.expert_epochs}",
                f"base_train_epochs={args.base_epochs}",
                f"ft_epochs={args.ft_epochs}",
                f"num_workers={args.num_workers}",
                f"eval_num_workers={args.eval_num_workers}",
                f"device='{args.device}'",
            ]
        )
    elif args.device:
        pieces.append(f"device='{args.device}'")
    if args.extra_override:
        pieces.extend(parse_csv(args.extra_override.replace("|", ",")))
    return "|".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "medium", "large", "paper"), default="quick")
    parser.add_argument("--datasets", default="mnist,cifar10,cifar100,webvision")
    parser.add_argument("--run-keys", default="", help="Optional comma-separated run keys, for example mnist_sym20,cifar10_asym40")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--expert-epochs", type=int, default=1)
    parser.add_argument("--base-epochs", type=int, default=1)
    parser.add_argument("--ft-epochs", type=int, default=1)
    parser.add_argument("--extra-override", default="", help="Additional overrides separated by |")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    datasets = set(parse_csv(args.datasets))
    run_keys = set(parse_csv(args.run_keys))
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    log_root = args.log_root or DEFAULT_CHECKPOINT_ROOT / f"{args.suite}_matrix"
    log_root.mkdir(parents=True, exist_ok=True)

    selected = [exp for exp in EXPERIMENTS if exp.dataset in datasets and (not run_keys or exp.run_key in run_keys)]
    if not selected:
        raise SystemExit(f"No experiments selected for datasets={sorted(datasets)}")

    for exp in selected:
        for seed in seeds:
            run_id = f"{args.suite}_{exp.run_key}_seed{seed}"
            run_dir = run_dir_for(log_root, run_id)
            if args.skip_existing and has_result(run_dir):
                print(f"[skip] {run_id} already has event/checkpoint")
                continue

            override = build_override(exp, args)
            cmd = [
                str(args.python),
                "main.py",
                "--log-dir",
                str(log_root),
                "--c",
                exp.config,
                "--e",
                exp.episode,
                "--random_seed",
                str(seed),
                "--override",
                override,
            ]
            env = os.environ.copy()
            env["SLURM_JOB_ID"] = run_id
            print(" ".join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
