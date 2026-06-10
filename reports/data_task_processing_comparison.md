# BCI Data Task Processing Comparison

Last updated: 2026-06-05

This report compares how different BCI data types and EEG tasks were processed locally. All pipelines are CPU-only and are intended as transparent baselines, not optimized final models.

## 1. Scripts

| Script | Data/tasks |
| --- | --- |
| `scripts/run_ssvep_feature_pipeline.py` | EEG SSVEP, Tsinghua S1 |
| `scripts/run_additional_bci_pipelines.py` | EEG MI/ME, EEG P300, fNIRS |
| `scripts/run_emg_fmri_pipelines.py` | EMG gestures, auditory fMRI |

Run:

```bash
python3 -B scripts/run_ssvep_feature_pipeline.py
python3 -B scripts/run_additional_bci_pipelines.py
python3 -B scripts/run_emg_fmri_pipelines.py
```

Data sources added for the EMG/fMRI extension:

| Local data | Source |
| --- | --- |
| `data/raw/emg_gestures/emg_data_for_gestures.retry.zip` | UCI EMG Data for Gestures: <https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures> |
| `data/raw/fmri_moae/MoAEpilot.bids.zip` | SPM/UCL MoAEpilot BIDS fMRI: <https://www.fil.ion.ucl.ac.uk/spm/download/data/MoAEpilot/MoAEpilot.bids.zip> |

## 2. High-Level Difference

| Modality/task | Signal pattern | Preprocessing target | Feature extraction | Decode sanity check |
| --- | --- | --- | --- | --- |
| EEG SSVEP | Frequency-locked visual response | posterior channels + flicker window + filter bank | CCA/FBCCA scores against sine/cosine frequency templates | argmax over 40 target scores |
| EEG MI/ME | Mu/beta rhythm modulation over sensorimotor cortex | motor channels + cue-aligned epochs + 8-30 Hz | filter-bank log variance | leave-one-out nearest centroid |
| EEG P300 | Event-related positive component around 300 ms | flash-aligned epochs + baseline + 0.1-30 Hz | temporal window means over ERP epochs | leave-one-out nearest centroid |
| fNIRS | Slow hemodynamic HbO/HbR changes | long task epochs + 0.01-0.09 Hz + baseline | task-window mean HbO/HbR features | leave-one-out nearest centroid |
| EMG gestures | Peripheral muscle electrical activity | 20-450 Hz + nonzero gesture windows | RMS/MAV/waveform length/zero-crossing/slope-change | nearest centroid |
| fMRI auditory | Slow BOLD voxel time series | brain mask + percent-signal-change + event labels | grid ROI time series, block means, HRF-GLM beta/t | weak nearest centroid; GLM contrast is the meaningful sanity output |

## 3. EEG SSVEP

Output directory:

```text
data/processed/ssvep_s1_pipeline/
```

Pipeline:

1. Raw Tsinghua `S1.mat`: `data` shape `(64, 1500, 40, 6)`.
2. Expand to trials and select posterior visual channels: `(240, 9, 1500)`.
3. Keep 5 s flicker window: `(240, 9, 1250)`.
4. Apply five filter banks: `(240, 5, 9, 1250)`.
5. Compute CCA correlations for 40 target frequency templates: `(240, 5, 40)`.
6. Weighted FBCCA features: `(240, 40)`.

Files:

| File | Meaning |
| --- | --- |
| `step2_occipital_epochs.npz` | selected posterior EEG epochs |
| `step3_stimulation_window.npz` | 5 s visual stimulation windows |
| `step4_filterbank_epochs.npz` | filter-bank EEG |
| `step5_fbcca_features.npz` | CCA correlations and final FBCCA scores |
| `step6_argmax_decode_sanity.csv` | direct argmax predictions |

Sanity result:

```text
238 / 240 correct
accuracy = 0.9917
```

Interpretation: SSVEP has explicit frequency templates, so CCA/FBCCA can directly decode without ML/DL.

## 4. EEG MI/ME: Same EEG Modality, Different Motor Tasks

Output directory:

```text
data/processed/mi_eeg_s001_pipeline/
```

Input:

```text
PhysioNet EEGMMI S001R03/R04/R06 EDF
```

Tasks processed:

| Run | Task | Labels |
| --- | --- | --- |
| `S001R03.edf` | executed left/right fist | `left_fist`, `right_fist` |
| `S001R04.edf` | imagined left/right fist | `left_fist`, `right_fist` |
| `S001R06.edf` | imagined both fists/both feet | `both_fists`, `both_feet` |

Pipeline:

1. Read EDF signal and annotations.
2. Select 15 sensorimotor channels:
   `FC3, FC1, FCZ, FC2, FC4, C3, C1, CZ, C2, C4, CP3, CP1, CPZ, CP2, CP4`.
3. Bandpass 8-30 Hz.
4. Epoch each non-rest cue from 0.5 s to 4.0 s.
5. Compute log variance features over four bands:
   `8-12`, `12-16`, `16-24`, `24-30` Hz.

Outputs:

| File | Shape | Meaning |
| --- | ---: | --- |
| `step_mi_epochs_8_30hz.npz` | `(45, 15, 560)` | motor EEG epochs |
| `step_mi_logvar_features.npz` | `(45, 60)` | 15 channels x 4 bands log-variance features |
| `labels.csv` | 45 rows | task/run/label metadata |
| `decode_sanity_by_task.csv` | 45 rows | nearest-centroid predictions |

Sanity results:

| Task | Accuracy | Balanced accuracy |
| --- | ---: | ---: |
| executed left/right fist | 0.4667 | 0.4643 |
| imagined left/right fist | 0.6000 | 0.5982 |
| imagined both fists/both feet | 0.7333 | 0.7321 |

Interpretation: MI/ME has no known sine template like SSVEP. Features are statistical rhythm features; stronger methods usually use CSP, Riemannian geometry, EEGNet, or subject adaptation.

## 5. EEG P300: Same EEG Modality, Event-Related Task

Output directory:

```text
data/processed/p300_sub01_pipeline/
```

Input:

```text
OpenNeuro ds003190 sub-01 ses-01 task-cnos run-4 BrainVision files
```

Pipeline:

1. Read BrainVision float32 EEG.
2. Use 8 EEG channels: `FZ, Cz, P3, PZ, P4, PO7, PO8, Oz`.
3. Bandpass 0.1-30 Hz.
4. Epoch every flash event from -0.2 s to 0.8 s.
5. Baseline correct with -0.2 to 0 s.
6. Extract mean amplitude in windows:
   `0-0.2`, `0.2-0.4`, `0.4-0.6`, `0.6-0.8` s.

Outputs:

| File | Shape | Meaning |
| --- | ---: | --- |
| `step_p300_epochs_0p1_30hz.npz` | `(1189, 8, 256)` | ERP epochs |
| `step_p300_window_mean_features.npz` | `(1189, 32)` | 8 channels x 4 time-window means |
| `decode_sanity.csv` | 1189 rows | target/non-target sanity predictions |

Sanity result:

```text
accuracy = 0.5114
balanced_accuracy = 0.4931
target accuracy = 0.4564
non_target accuracy = 0.5297
```

Interpretation: This simple feature is intentionally minimal and not a strong P300 decoder. Better P300 pipelines use xDAWN, SWLDA/RLDA, temporal-spatial filtering, trial averaging, or sequence-level decoding.

## 6. fNIRS: Different Modality, Hemodynamic Task

Output directory:

```text
data/processed/fnirs_subject01_pipeline/
```

Input:

```text
Shin hybrid EEG-NIRS subject_01.mat
```

Labels:

```text
Mental Arithmetic
Motor Imagery
Idle State
```

Pipeline:

1. Load 32 HbO/HbR concentration channels.
2. Bandpass 0.01-0.09 Hz for slow hemodynamic changes.
3. Epoch from -5 s to 25 s around each task marker.
4. Baseline correct using -1 to 0 s.
5. Extract mean values in 5-10 s and 10-15 s task windows.

Outputs:

| File | Shape | Meaning |
| --- | ---: | --- |
| `step_fnirs_epochs_0p01_0p09hz.npz` | `(90, 32, 400)` | filtered HbO/HbR epochs |
| `step_fnirs_window_mean_features.npz` | `(90, 64)` | 32 channels x 2 task-window means |
| `decode_sanity.csv` | 90 rows | 3-class sanity predictions |

Sanity result:

```text
accuracy = 0.5778
balanced_accuracy = 0.5778
```

Interpretation: fNIRS is not frequency-template decoding. It uses slow blood oxygenation responses, long windows, and mean/slope-like hemodynamic features.

## 7. EMG Gestures: Not Brain, But Common in Hybrid BCI/HMI

Output directory:

```text
data/processed/emg_gestures_pipeline/
```

Input:

```text
UCI EMG Data for Gestures, subject 01, two recordings
```

Pipeline:

1. Read tab-separated raw data with columns:
   `time`, `channel1`-`channel8`, `class`.
2. Estimate sampling rate from millisecond timestamps: `1000 Hz`.
3. Bandpass filter each recording: `20-450 Hz`.
4. Ignore `class=0` unmarked intervals and split gesture segments into 200 ms windows with 100 ms stride.
5. Extract time-domain EMG features:
   RMS, MAV, waveform length, zero crossings, slope sign changes.

Outputs:

| File | Shape | Meaning |
| --- | ---: | --- |
| `step1_emg_raw_subject01.npz` | `(121170, 8)` | raw 8-channel EMG samples |
| `step2_emg_bandpass_20_450hz.npz` | `(121170, 8)` | filtered EMG |
| `step3_emg_windows_200ms.npz` | `(388, 8, 200)` | 200 ms gesture windows |
| `step4_emg_time_domain_features.npz` | `(388, 40)` | 8 channels x 5 features |
| `labels.csv` | 388 rows | window labels and file/segment metadata |

Ground truth:

```text
class column in the raw UCI file
0 = unmarked and not used for gesture-window decoding
1-6 = gesture classes used here
```

Sanity results:

| Check | Accuracy | Balanced accuracy |
| --- | ---: | ---: |
| Leave-one-window-out | 0.8737 | 0.8725 |
| Train recording 1 -> test recording 2 | 0.7861 | 0.7839 |

Interpretation: EMG is not EEG and not brain activity. It measures muscle electrical activity from peripheral electrodes. It is still relevant for BCI/HMI because many systems fuse EEG intent with EMG execution or use EMG as an auxiliary control signal.

## 8. fMRI Auditory: Voxel BOLD Time Series and Task GLM

Output directory:

```text
data/processed/fmri_moae_pipeline/
```

Input:

```text
SPM/UCL MoAEpilot BIDS auditory fMRI dataset
```

Pipeline:

1. Read BIDS NIfTI BOLD file: `(64, 64, 64, 84)`, TR = 7 s.
2. Create a simple intensity brain mask: `(64, 64, 64)`, 157342 voxels.
3. Convert masked voxel time series to percent signal change and linear-detrend.
4. Average voxels into a `3 x 3 x 3` spatial grid: ROI time series `(84, 27)`.
5. Convert BIDS events into per-volume labels:
   `42 listening` and `42 rest` volumes.
6. Aggregate each 42 s block into block features: `(14, 27)`.
7. Fit a lightweight ROI GLM using intercept, linear drift, and HRF-convolved listening regressor.

Outputs:

| File | Shape | Meaning |
| --- | ---: | --- |
| `step1_nifti_metadata.json` | metadata | NIfTI/BIDS header information |
| `step2_brain_mask_and_mean.npz` | mask `(64,64,64)` | mean image, mask, voxel coordinates |
| `step3_fmri_roi_timeseries_psc.npz` | `(84, 27)` | ROI percent-signal-change time series |
| `volume_labels.csv` | 84 rows | per-volume `rest/listening` labels |
| `step4_fmri_block_features.npz` | `(14, 27)` | 42 s block-level ROI features |
| `step5_fmri_listening_minus_rest_contrast.npz` | `(27,)` | ROI listening-minus-rest contrast |
| `step6_fmri_roi_glm_features.npz` | betas `(3,27)`, t `(27,)` | GLM task beta/t features |

Ground truth:

```text
BIDS events.tsv explicitly marks listening blocks.
Rest blocks are implicit intervals outside listening events.
```

Sanity result:

```text
block nearest-centroid accuracy = 0.2143
```

This low number is useful: it shows that fMRI should not be treated like SSVEP argmax decoding. The meaningful fMRI baseline is usually event/block modeling with an HRF-convolved GLM, motion correction, normalization, smoothing, and statistical contrast maps. This local script only runs the CPU-light feature extraction needed to inspect the data shape and task labels.

## 9. What This Shows

Same modality can use very different processing depending on task:

```text
EEG + SSVEP -> frequency template matching
EEG + MI    -> rhythm band-power/log-variance features
EEG + P300  -> time-locked ERP waveform features
```

Different modality changes the physiological signal:

```text
EEG   -> fast electrical activity, millisecond scale
fNIRS -> slow HbO/HbR hemodynamics, seconds scale
EMG   -> muscle electrical activity, millisecond scale, not brain
fMRI  -> voxel BOLD hemodynamics, seconds scale, spatially rich
```

So "BCI preprocessing" is not one universal recipe. The pipeline is driven by:

1. acquisition modality,
2. task paradigm,
3. expected neural/physiological response,
4. decoder type.
