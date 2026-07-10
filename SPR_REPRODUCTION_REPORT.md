# SPR Reproduction Report

Date: 2026-07-10

Paper: Continual Learning on Noisy Data Streams via Self-Purified Replay, ICCV 2021.

Local repo: `/home/undefined/Desktop/bci/papers/CL_TTA_security/CL_defenses/spr_official`

## Environment

- Python: `/home/undefined/Disk/ai-storage/.venv-rttdp-py39/bin/python`
- Python version: 3.9.25
- PyTorch: 2.3.0+cu121
- Torchvision: 0.18.0+cu121
- CUDA: available
- GPU: NVIDIA GeForce RTX 4070 SUPER, 12 GB
- Reused Disk caches/env:
  - virtualenv under `/home/undefined/Disk/ai-storage/.venv-rttdp-py39`
  - pip/uv caches under `/home/undefined/Disk/ai-storage`
  - checkpoints under `/home/undefined/Disk/ai-storage/SPR_checkpoints`
  - data under `/home/undefined/Disk/ai-storage/SPR_data`

Additional packages installed in the reused env:

- `tensorboardX`
- `colorful`
- `colorlog`
- `kornia==0.4.1`

## Data Status

SPR uses `spr_official/data -> /home/undefined/Disk/ai-storage/SPR_data`.

Available:

- MNIST: `/home/undefined/Disk/ai-storage/SPR_data/mnist/MNIST`
- CIFAR-10: `/home/undefined/Disk/ai-storage/SPR_data/cifar10`, symlinked to `/home/undefined/Desktop/bci/code/sfda/2025CVPR-source-free-unlearning/data`

Missing / not a matching training dataset:

- CIFAR-100: only `CIFAR-100-C` was found under Disk; this is not the original CIFAR-100 training set expected by torchvision.
- WebVision: expected `data/webvision/info/synsets.txt`, `train_filelist_google.txt`, `val_filelist.txt`, plus image files. This directory was not found.

## Compatibility Patches

The official code targets older PyTorch/networkx versions. These local patches were required for this machine:

- `utils.py`: shimmed removed `torch.solve` through `torch.linalg.solve` before importing `kornia==0.4.1`.
- `models/SPR.py`: replaced `nx.from_numpy_matrix` with `nx.from_numpy_array`.
- `models/SPR.py`: moved delayed-buffer corruption flags to the label tensor device before GPU indexing.

## Added Utilities

- `run_spr_experiments.py`: runs the paper or quick experiment matrix with stable `SLURM_JOB_ID` names.
- `collect_spr_results.py`: parses TensorBoard event files and compares final overall accuracy with the paper table values.

Examples:

```bash
/home/undefined/Disk/ai-storage/.venv-rttdp-py39/bin/python run_spr_experiments.py \
  --suite quick --datasets cifar10 --run-keys cifar10_sym20 --seeds 1

/home/undefined/Disk/ai-storage/.venv-rttdp-py39/bin/python run_spr_experiments.py \
  --suite paper --datasets mnist,cifar10 --seeds 1,2,3,4,5

/home/undefined/Disk/ai-storage/.venv-rttdp-py39/bin/python collect_spr_results.py \
  --roots /home/undefined/Disk/ai-storage/SPR_checkpoints/large200_matrix \
          /home/undefined/Disk/ai-storage/SPR_checkpoints/large_matrix \
          /home/undefined/Disk/ai-storage/SPR_checkpoints/medium_matrix \
          /home/undefined/Disk/ai-storage/SPR_checkpoints/quick_matrix \
          /home/undefined/Disk/ai-storage/SPR_checkpoints/short_runs \
  --csv /home/undefined/Disk/ai-storage/SPR_checkpoints/spr_results.csv
```

## Completed Local Runs

The completed runs use symmetric 20% label noise and seed 1. They are
single-seed budget studies, while the paper reports the mean over five seeds.
All local runs use `num_workers=0` and `eval_num_workers=0`.

| Dataset | Expert / base / FT epochs | Local accuracy | Paper accuracy | Delta |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 1 / 1 / 1 | 70.18 | 85.40 | -15.22 |
| MNIST | 10 / 10 / 5 | 76.21 | 85.40 | -9.19 |
| MNIST | 50 / 50 / 10 | 77.43 | 85.40 | -7.97 |
| CIFAR-10 | 1 / 1 / 1 | 16.01 | 43.90 | -27.89 |
| CIFAR-10 | 10 / 10 / 5 | 22.18 | 43.90 | -21.72 |
| CIFAR-10 | 50 / 50 / 10 | 34.61 | 43.90 | -9.29 |
| CIFAR-10 | 200 / 200 / 20 | 37.77 | 43.90 | -6.13 |

The CIFAR-10 200 / 200 / 20 run took 9,102.8 seconds (2 h 31 min 43 s)
on the local RTX 4070 SUPER. Increasing the main training budget from 50 to
200 epochs improved accuracy by 3.16 percentage points. This is still below
the paper-default budget of 4000 / 3000 / 50 and does not yet reproduce the
paper's five-seed result.

Result CSV:

`/home/undefined/Disk/ai-storage/SPR_checkpoints/spr_results.csv`

Run artifacts:

- `/home/undefined/Disk/ai-storage/SPR_checkpoints/short_runs/short_mnist_sym20_e1`
- `/home/undefined/Disk/ai-storage/SPR_checkpoints/medium_matrix`
- `/home/undefined/Disk/ai-storage/SPR_checkpoints/large_matrix`
- `/home/undefined/Disk/ai-storage/SPR_checkpoints/large200_matrix`
- `/home/undefined/Disk/ai-storage/SPR_checkpoints/quick_matrix/quick_cifar10_sym20_seed1`

## Paper Reference Values

Table 1, final overall accuracy:

| Dataset | Noise | SPR paper accuracy |
| --- | --- | ---: |
| MNIST | symmetric 20% | 85.4 |
| MNIST | symmetric 40% | 86.7 |
| MNIST | symmetric 60% | 84.8 |
| MNIST | asymmetric 20% | 86.8 |
| MNIST | asymmetric 40% | 86.0 |
| CIFAR-10 | symmetric 20% | 43.9 |
| CIFAR-10 | symmetric 40% | 43.0 |
| CIFAR-10 | symmetric 60% | 40.0 |
| CIFAR-10 | asymmetric 20% | 44.5 |
| CIFAR-10 | asymmetric 40% | 43.9 |
| WebVision | real noise | 40.0 |

Table 2, CIFAR-100 final overall accuracy:

| Dataset | Noise | SPR paper accuracy |
| --- | --- | ---: |
| CIFAR-100 | random symmetric 20% | 21.5 |
| CIFAR-100 | random symmetric 40% | 21.1 |
| CIFAR-100 | random symmetric 60% | 18.1 |
| CIFAR-100 | superclass symmetric 20% | 20.5 |
| CIFAR-100 | superclass symmetric 40% | 19.8 |
| CIFAR-100 | superclass symmetric 60% | 16.5 |

## Full Paper Matrix

The full matrix generated by `run_spr_experiments.py --suite paper --datasets mnist,cifar10,cifar100,webvision --seeds 1,2,3,4,5 --dry-run` contains 85 commands:

- MNIST: 5 noise settings x 5 seeds
- CIFAR-10: 5 noise settings x 5 seeds
- CIFAR-100: 6 noise settings x 5 seeds
- WebVision: 1 real-noise setting x 5 seeds

The official default training is expensive:

- `expert_train_epochs=4000`
- `base_train_epochs=3000`
- `ft_epochs=50`

These epochs are triggered repeatedly as delayed buffers fill. Extrapolating
from the measured CIFAR-10 200 / 200 / 20 run, one paper-default CIFAR-10 run
would require roughly 40 hours on this machine, before accounting for software
and workload variability. The complete 85-command matrix is therefore a
multi-week single-GPU workload. CIFAR-100 and WebVision also cannot start until
their expected original datasets are provided.
