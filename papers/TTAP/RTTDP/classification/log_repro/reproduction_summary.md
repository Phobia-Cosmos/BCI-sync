# RTTDP Reproduction Summary

Date: 2026-07-04

## Environment

- Python: 3.9.25
- PyTorch: 2.3.0+cu121
- torchvision: 0.18.0+cu121
- torchaudio: 2.3.0+cu121
- GPU: NVIDIA GeForce RTX 4070 SUPER, 12GB
- venv: `/home/undefined/Disk/ai-storage/.venv-rttdp-py39`

## Data

- CIFAR-100-C:
  - directory: `/home/undefined/Disk/ai-storage/RTTDP/data/CIFAR-100-C`
  - tar: `/home/undefined/Disk/ai-storage/RTTDP/data/CIFAR-100-C.tar`
  - tar size: 2,918,473,216 bytes
- ImageNet-C partial:
  - source: Hugging Face `WNJXYK/TTA-ImageNet-C`
  - prepared subset: `data/ImageNet-C/gaussian_noise/5`
  - list: `robustbench/data/imagenet_test_image_ids_hf_gaussian_noise_5_current.txt`
  - samples/classes: 218/218

## CIFAR-100-C Full Runs

All CIFAR-100-C runs use severity 5, all 15 corruptions, seed 1, ResNeXt-29/AugMix.

| Setting | Local mean error | Paper Table 3 value | Difference |
| --- | ---: | ---: | ---: |
| Source | 46.45 | 46.23 / 46.33 | +0.22 / +0.12 |
| TENT, RTTDP NoAttack, uniform 50% stream | 60.52 | 60.25 | +0.27 |
| TENT, RTTDP BLE + feature consistency, uniform 50% stream | 69.45 | 73.93 | -4.48 |
| TENT, RTTDP NHE + feature consistency, uniform 50% stream | 92.03 | 92.08 | -0.05 |

Key logs:

- `log_repro/cifar100_source_seed1_full_260704_135137.log`
- `log_repro/cifar100_tent_rttdp_noattack_seed1_full_260704_135422.log`
- `log_repro/cifar100_tent_uniform_ble_seed1_full_260704_181402.log`
- `log_repro/cifar100_tent_uniform_nhe_seed1_full_260704_114706.log`

## ImageNet-C Partial Runs

These are not full Table 4 reproductions. They use only `gaussian_noise`, severity 5, a 218-sample partial HF subset.

| Setting | Local error |
| --- | ---: |
| ResNet-50 source | 97.25 |
| clean TENT | 83.49 |
| TENT, RTTDP NHE + feature consistency, uniform 50% stream | 86.46 |

Key logs:

- `log_repro/imagenet_c_gaussian_current_source_260704_134838.log`
- `log_repro/imagenet_c_gaussian_current_tent_clean_260704_134915.log`
- `log_repro/imagenet_c_gaussian_current_tent_nhe_260704_134935.log`

## Notes

- Full ImageNet-C requires the standard 15 corruptions at severity 5. The compressed Zenodo set for these corruptions is roughly 50GB, which does not fit comfortably in the current remaining disk space.
- The Hugging Face parquet source allows partial ImageNet-C preparation without downloading the full Zenodo archives, but streaming was unstable and slow in this environment.
- `RTTDP_IMAGENETC_LIST` can be set to use a custom ImageNet-C sample list without changing default repo behavior.
