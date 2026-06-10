# SFDA Code Run Report

Last updated: 2026-06-07

This report records the local code inspection and runnable smoke tests for:

1. `2025ICLR-SPDIM: Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG`
2. `2025AAAI-Personalized sleep staging leveraging source free unsupervised domain adaptation`

## 1. Repositories Checked

| Method/paper | Local path | Source | Status |
| --- | --- | --- | --- |
| SF-UIDA sleep staging | `code/SF-UIDA` | <https://github.com/xiaobaben/SF-UIDA> | Official repo cloned, commit `4172de4` |
| TSMNet/SPDDSBN method family used by SPDIM | `code/TSMNet` | <https://github.com/rkobler/TSMNet> | Related repo cloned, commit `90293b9` |
| SPDIM exact implementation | not found | arXiv/OpenReview search | No verified official public repo found |
| Source-free EEG visual adaptation | `code/Deep-BCI/1_Intelligent_BCI/Source_Free_Subject_Adaptation_for_EEG` | already local | Related EEG SFDA repo, requires visual EEG dataset |
| SSVEP SFDA | `code/SFDA-SSVEP-BCI` | already local | MATLAB repo, requires Benchmark/BETA SSVEP datasets |

## 2. Official Entry Points Tried

Commands:

```bash
python3 -B code/SF-UIDA/Subject_UDA/main.py
python3 -B code/TSMNet/experiments/main.py
```

Results:

| Repo | Stop point |
| --- | --- |
| `code/SF-UIDA` | `ModuleNotFoundError: No module named 'torch'` |
| `code/TSMNet` | `ModuleNotFoundError: No module named 'hydra'` |

The current environment has `numpy/scipy/pandas`, but not `torch`, `sklearn`, `mne`, `moabb`, `hydra`, `omegaconf`, `skorch`, or `geoopt`.

Full official runs were not launched because the dependencies and large domain datasets are missing. This is not only an installation issue: `SF-UIDA` also leaves dataset paths empty in `get_filepath()`, and expects preprocessed sleep `.npy` files plus saved source-model checkpoints when `pretrain=False`.

## 3. SF-UIDA: What The Official Code Does

Official entry:

```text
code/SF-UIDA/Subject_UDA/main.py
```

Data layout expected by `dataloader/dataloader.py`:

```text
{filepath}/{subject}/data/{sequence_id}.npy
{filepath}/{subject}/label/{sequence_id}.npy
```

Each `.npy` data file is a sleep sequence, not one isolated epoch. The code assumes:

```text
x_data shape: (20, channels, 3000)
y_data shape: (20,)
```

For `ISRUC`:

```text
EOG = x_data[:, :2, :]
EEG = x_data[:, 2:, :]
```

For `SleepEDF`:

```text
EEG = x_data[:, 0:2, :]
EOG = x_data[:, 2, :].view(20, 1, -1)
```

So this is not just data preprocessing. The method performs full model adaptation and sleep-stage decoding:

1. Source pretraining on labeled source subjects.
2. Target subject is treated as one individual target domain.
3. Source model predicts target subject before adaptation.
4. Subject-specific adaptation uses unlabeled target EEG/EOG only.
5. Sequential cross-view contrast/adaptation step updates feature extractor and encoder.
6. Teacher model generates confident pseudo-labels.
7. Pseudo-label fine-tuning updates the personalized model.
8. Ground-truth target labels are used only for final evaluation.

The adaptation code is in:

```text
code/SF-UIDA/Subject_UDA/model/algorithm.py
code/SF-UIDA/Subject_UDA/trainer/trainer.py
```

## 4. SPDIM / TSMNet: What The Code Family Does

No verified public official SPDIM repository was found. The closest related open code is `TSMNet`, which is the SPD manifold / SPD domain-specific batch-norm code family referenced by the SPDIM line of work.

Official TSMNet run pattern:

```bash
cd code/TSMNet/experiments
python main.py dataset=bnci2014001
python main.py evaluation=inter-subject+uda dataset=bnci2014001
```

The repo expects a conda environment from:

```text
code/TSMNet/environment.yaml
```

It downloads/preprocesses MOABB datasets through `mne` and `moabb`.

TSMNet data/model flow:

```text
MOABB EEG epochs
-> temporal CNN
-> spatial CNN
-> covariance pooling
-> SPD manifold layers: BiMap / ReEig
-> SPD batch norm or SPD domain-specific batch norm
-> LogEig tangent-space feature
-> linear classifier
-> prediction
```

SPDIM conceptually extends this family. After target marginal alignment, it learns a small target-domain SPD bias parameter with information maximization. That means SPDIM is doing feature extraction, source decoding, and target-domain adaptation, not only preprocessing.

## 5. Local CPU Smoke Runs

Smoke script:

```bash
python3 -B scripts/run_sfda_smoke_tests.py
```

Output root:

```text
data/processed/sfda_smoke/
```

These are dependency-free local demonstrations. They are not official reproduction numbers.

### 5.1 SPDIM/TSM-like MI EEG Run

Output:

```text
data/processed/sfda_smoke/spdim_like_mi/
```

Input:

```text
data/processed/mi_eeg_s001_pipeline/step_mi_epochs_8_30hz.npz
```

Flow:

1. Use `executed_left_right_fist` as labeled source domain.
2. Use `imagined_left_right_fist` as unlabeled target domain.
3. Convert each EEG epoch `(15, 560)` into an SPD covariance matrix.
4. Apply matrix logarithm and vectorize upper triangle.
5. Train a source nearest-centroid classifier in log-covariance space.
6. Adapt target features with log-Euclidean recentering.
7. Fit a tiny information-maximization output bias using target predictions only.
8. Use target labels only after adaptation for sanity metrics.

Key output:

```text
step1_mi_covariance_log_features.npz
features shape = (45, 120)
```

Results:

| Target case | Source only | Recentered | Recentered + IM bias |
| --- | ---: | ---: | ---: |
| Full imagined target | 0.5333 | 0.7333 | 0.7333 |
| Artificial label-shift target | 0.5455 | 0.8182 | 0.8182 |

The IM bias barely moved because the target predictions were already close to balanced in this tiny binary example. The important part is the data flow: source model state is fixed, target labels are hidden, and target features are adapted source-free.

### 5.2 SF-UIDA Sleep Shape Demo

Output:

```text
data/processed/sfda_smoke/sf_uida_sleep_shape_demo/
```

Generated official-style sample:

```text
ISRUC/1/data/0.npy   shape = (20, 8, 3000)
ISRUC/1/label/0.npy  shape = (20,)
```

This demonstrates what the official SF-UIDA dataloader expects before the PyTorch model starts.

### 5.3 SF-UIDA-like Toy Personalization Run

Output:

```text
data/processed/sfda_smoke/sf_uida_like_toy/
```

Flow:

1. Create source sleep-stage features and one shifted target individual.
2. Store source centroids as the source model state.
3. Source-only predict target.
4. Stage 1 aligns target marginal feature distribution.
5. Stage 2 keeps confident pseudo-labels and updates personalized prototypes.
6. Target labels are used only for final sanity metrics.

Results:

| Stage | Accuracy | Balanced accuracy |
| --- | ---: | ---: |
| Source only | 0.9500 | 0.9629 |
| Stage 1 marginal alignment | 0.9800 | 0.9886 |
| Stage 2 pseudo-label personalization | 0.9800 | 0.9886 |

Pseudo-labeled samples used:

```text
71 / 100
```

Stage 2 did not improve beyond Stage 1 in this toy run because Stage 1 already fixed most errors, but it shows exactly where confident pseudo-labels enter the SF-UIDA training logic.

## 6. Practical Run Order

For learning and debugging, use this order:

1. Read `scripts/run_sfda_smoke_tests.py`.
2. Inspect `data/processed/sfda_smoke/spdim_like_mi/summary.json`.
3. Inspect `code/SF-UIDA/Subject_UDA/dataloader/dataloader.py`.
4. Inspect `code/SF-UIDA/Subject_UDA/model/algorithm.py`.
5. Inspect `code/SF-UIDA/Subject_UDA/trainer/trainer.py`.
6. Inspect `code/TSMNet/spdnets/models/tsmnet.py`.
7. Inspect `code/TSMNet/spdnets/batchnorm.py`.
8. Only then set up the heavy official environments and datasets.

## 7. What To Install For Full Official Runs

For `SF-UIDA`:

```text
torch
scikit-learn
numpy
preprocessed SleepEDF/ISRUC/HMC .npy files
source model checkpoints, unless running pretrain=True first
```

For `TSMNet`:

```text
conda env create --file code/TSMNet/environment.yaml --prefix code/TSMNet/venv
```

This includes old pinned versions:

```text
python 3.8
pytorch 1.9
scikit-learn 1.0
mne 0.22
moabb 0.4
hydra-core 1.1
pyriemann 0.2
geoopt pinned from GitHub
```

Because those versions are old and the official experiments auto-download public EEG datasets, running the full stack should be done in an isolated conda environment, not the current base Python.
