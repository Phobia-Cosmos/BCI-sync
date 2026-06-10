# BCI / SFDA Paper, Code, and Data Inventory

Last updated: 2026-06-04

## 1. Local Paper Inventory

Current workspace already contained 29 PDFs under the project root and `others/`. I added the following BCI/SFDA-related PDFs under `papers/sfda/`:

| Paper | Local PDF | Code / status |
| --- | --- | --- |
| Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces | `papers/sfda/2023Arxiv-Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces.pdf` | `code/SFDA-SSVEP-BCI`, official MATLAB repo |
| Source-free Subject Adaptation for EEG-based Visual Recognition | `papers/sfda/2023Arxiv-Source-free Subject Adaptation for EEG-based Visual Recognition.pdf` | `code/Deep-BCI/1_Intelligent_BCI/Source_Free_Subject_Adaptation_for_EEG`, sparse checkout |
| SPDIM: Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG | `papers/sfda/2026Arxiv-SPDIM Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG.pdf` | no public code found in quick search |
| Lightweight Source-Free Domain Adaptation based on Adaptive Euclidean Alignment for BCIs | `papers/sfda/2025JBHI-Lightweight Source-Free Domain Adaptation based on Adaptive Euclidean Alignment for BCIs.pdf` | no official repo found; useful to implement on top of SHOT/NRC/MOABB |
| Prediction Consistency and Confidence-Based Proxy Domain Construction for Cross-Subject EEG Classification | `papers/sfda/2025JBHI-Prediction Consistency and Confidence-Based Proxy Domain Construction for Cross-Subject EEG Classification.pdf` | `code/PDCC`, official repo |

Existing local papers that are directly relevant:

| Existing local PDF | Code / source |
| --- | --- |
| `2026AAAI-Probability Distribution Alignment and Low-Rank Weight Decomposition for Source-Free Domain Adaptive Brain Decoding.pdf` | no official repo found; closest executable baseline is `code/mindbridge` |
| `2020ICML-Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation.pdf` | `code/SHOT` |
| `others/2024CVPR-Source-Free Domain Adaptation with Frozen Multimodal Foundation Model.pdf` | `code/source-free-domain-adaptation`; also `code/DIFO-Plus` |
| `2024CVPR-MindBridge: A Cross-Subject Brain Decoding Framework.pdf` | `code/mindbridge` |
| `2025ICLR-CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding.pdf` | public repo: `https://github.com/wjq-learning/CBraMod`; not cloned because it is not SFDA-core |
| `2026AAAI-EEG Agent: A Unified Framework for Automated EEG Analysis Using Large Language Models.pdf` | public repo: `https://github.com/rebootingLine/EEGAgent`; not cloned because it is not SFDA-core |
| `2026AAAI-CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding.pdf` | public repo: `https://github.com/YifanZhuang/CAT-Net`; not cloned because it is not SFDA-core |

Known BCI/SFDA papers not fully downloaded:

| Paper | Reason |
| --- | --- |
| T-CMDP: Transformer-Based SFDA by Class-Balanced Multicentric Dynamic Pseudo-Labeling for Privacy-Preserving EEG-Based BCI Systems | MDPI page is open, but local PDF download returned HTTP 403 |
| Online Privacy-Preserving EEG Classification by Source-Free Transfer Learning | PubMed/IEEE metadata found; no official open PDF/code found |
| OSFTL-PL: Online source-free transfer learning with pseudo-labels for privacy-preserving EEG-based BCI | ScienceDirect metadata found; no official open PDF/code found |
| Lightweight Source-Free Transfer for Privacy-Perving Motor Imagery Classification | metadata found; no official open PDF/code found |

## 2. Local Code and Test Status

| Priority | Repo | Local path | Why read/test in this order | Current test status |
| --- | --- | --- | --- | --- |
| 1 | SHOT | `code/SHOT` | canonical SFDA baseline: source model + target-only adaptation | Python syntax check passed; full run blocked by missing `torch`, `torchvision`, `sklearn` |
| 2 | SFDA-SSVEP-BCI | `code/SFDA-SSVEP-BCI` | direct BCI+SFDA method and matches downloaded SSVEP data | MATLAB repo; no `matlab`/`octave` installed locally |
| 3 | Deep-BCI source-free EEG | `code/Deep-BCI/1_Intelligent_BCI/Source_Free_Subject_Adaptation_for_EEG` | EEG visual-recognition source-free adaptation; three-stage source/generator/target adaptation | sparse checkout done; Python syntax check passed; full run needs `torch` and EEG visual dataset |
| 4 | PDCC | `code/PDCC` | recent EEG source-free/proxy-domain method with official code | fixed one local f-string syntax bug in `get_proxy_domain.py`; syntax check passed; full run needs `sklearn` and benchmark EEG features |
| 5 | MindBridge | `code/mindbridge` | needed for the AAAI 2026 brain-decoding SFDA paper baseline | syntax check passed; full run needs NSD access, Hugging Face data/checkpoints, GPU, `torch` |
| 6 | DIFO / DIFO-Plus | `code/source-free-domain-adaptation`, `code/DIFO-Plus` | multimodal SFDA ideas useful for CLIP/fMRI-style brain decoding | syntax check passed; full run needs `torch`, CLIP deps, image DA datasets |

Executed lightweight test:

```bash
python3 -B -m compileall -q code/SHOT code/PDCC \
  code/Deep-BCI/1_Intelligent_BCI/Source_Free_Subject_Adaptation_for_EEG \
  code/source-free-domain-adaptation/src code/DIFO-Plus/src code/mindbridge
```

Missing local dependencies:

```text
torch=False, torchvision=False, sklearn=False, mne=False, matlab/octave not found
```

## 3. Downloaded Data

| Type | Local path | Amount downloaded | Source |
| --- | --- | --- | --- |
| Motor imagery EEG | `data/raw/physionet_eegmmi/` | PhysioNet EEGMMI S001 runs R03, R04, R06 | `https://physionet.org/content/eegmmidb/1.0.0/` |
| SSVEP EEG | `data/raw/tsinghua_ssvep/` | Tsinghua Benchmark S1 + frequency/phase/channel files | `https://bci.med.tsinghua.edu.cn/download.html` |
| P300 / ERP EEG | `data/raw/openneuro_p300/` | OpenNeuro ds003190 sub-01 ses-01 task-cnos run-4 | `https://openneuro.org/datasets/ds003190` and S3 raw files |
| fNIRS / hybrid EEG-NIRS | `data/raw/hybrid_eeg_fnirs/` | Figshare 9198932 full zip + extracted `subject_01.mat` | `https://figshare.com/articles/dataset/9198932` |

Generated ground-truth manifests:

| Manifest | Rows | Ground truth |
| --- | ---: | --- |
| `data/processed/manifests/physionet_eegmmi_S001_annotations.csv` | 90 | `rest`, `left_fist`, `right_fist`, `both_fists`, `both_feet` |
| `data/processed/manifests/tsinghua_ssvep_S1_trials.csv` | 240 | `target_1` to `target_40`, with frequency and phase |
| `data/processed/manifests/openneuro_ds003190_sub01_ses01_cnos_run4_p300_events.csv` | 1189 | `P300_target` / `P300_non_target` |
| `data/processed/manifests/shin_hybrid_eeg_nirs_subject_01_events.csv` | 90 | `Mental Arithmetic`, `Motor Imagery`, `Idle State` |

Manifest script:

```bash
python3 -B scripts/create_dataset_manifests.py
```

## 4. How to Process Each Dataset

### PhysioNet EEGMMI

Raw files are EDF. Use MNE when available:

1. Read EDF with `mne.io.read_raw_edf`.
2. Bandpass 8-30 Hz for motor imagery, optionally notch 60 Hz.
3. Extract annotations `T0/T1/T2`.
4. Epoch around each cue, typically `[0.5, 4.0]` seconds after cue onset.
5. For binary MI, drop `T0` rest and train on `T1/T2`.

Ground truth mapping used in manifest:

| Run | Task | T1 | T2 |
| --- | --- | --- | --- |
| R03 | executed left/right fist | left fist | right fist |
| R04 | imagined left/right fist | left fist | right fist |
| R06 | imagined both fists/both feet | both fists | both feet |

### Tsinghua SSVEP Benchmark

`S1.mat` contains `data` with shape `[64, 1500, 40, 6]`.

1. Load `data`, `Freq_Phase.mat`, and `64-channels.loc`.
2. Recommended channels for the cloned SSVEP-SFDA repo: `Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2`.
3. Data are already segmented and downsampled to 250 Hz.
4. Use the stimulation segment after the 0.5 s cue; many SSVEP pipelines also compensate visual latency.
5. Bandpass around SSVEP harmonics, e.g. 6-90 Hz, plus notch if needed.

Ground truth is target index 1-40. The semantic stimulus for each target is represented by `freqs[target]` and `phases[target]`.

### OpenNeuro ds003190 P300

Downloaded BrainVision files for `sub-01_ses-01_task-cnos_run-4`.

1. Read `.vhdr` with MNE BrainVision reader.
2. Use stimulus markers `1..9` as flash events.
3. Target symbol is encoded by markers `101..109`; for this run only symbols `1..4` appear.
4. Epoch each flash, e.g. `[-0.2, 0.8]` or `[-0.2, 1.0]` seconds.
5. Baseline correct using pre-stimulus interval.
6. Bandpass approximately 0.1-30 Hz.

Ground truth is binary: flash symbol equals current target symbol -> `P300_target`; otherwise `P300_non_target`.

### Shin Hybrid EEG-NIRS

Downloaded Figshare dataset and extracted `matdata/subject_01.mat`.

1. Load `subject_01.mat`.
2. Use `mrk.time` as milliseconds, `mrk.y` as one-hot labels, and `mrk.className` as label names.
3. NIRS channels are oxy/deoxy concentration signals with sampling frequency 13.333 Hz.
4. Tutorial uses bandpass 0.01-0.09 Hz, epoch `[-5, 25]` seconds, baseline `[-1, 0]` seconds.
5. Feature example: jumping means over `[5, 10]` and `[10, 15]` seconds.

Ground truth is one of `Mental Arithmetic`, `Motor Imagery`, `Idle State`.

## 5. Suggested Learning / Test Order

1. Start with `SHOT`: understand classifier freezing, information maximization, and pseudo-label prototypes.
2. Run/read `SFDA-SSVEP-BCI`: this is the closest paper-code-data match in the workspace.
3. Read `Deep-BCI` source-free EEG adaptation: useful for EEG visual/subject adaptation logic.
4. Test `PDCC` after installing `sklearn`: proxy-domain construction is closer to privacy-preserving EEG transfer.
5. Read AEA and SPDIM papers as algorithm extensions; implement only after the above baselines are clear.
6. Move to `MindBridge` and the AAAI 2026 brain decoding SFDA paper: this requires NSD/fMRI data and stronger GPU resources.
7. Read `DIFO/DIFO-Plus` if you want to extend brain decoding with CLIP/multimodal SFDA ideas.

## 6. Sources Used

- SHOT paper/code: `https://proceedings.mlr.press/v119/liang20a.html`, `https://github.com/tim-learn/SHOT`
- SSVEP-SFDA paper/code: `https://arxiv.org/abs/2305.17403`, `https://github.com/osmanberke/SFDA-SSVEP-BCI`
- Deep-BCI source-free EEG code: `https://github.com/DeepBCI/Deep-BCI`
- DIFO code: `https://github.com/tntek/source-free-domain-adaptation`
- DIFO-Plus code: `https://github.com/tntek/DIFO-Plus`
- PDCC code: `https://github.com/SunseaIU/PDCC`
- MindBridge code: `https://github.com/littlepure2333/MindBridge`
- PhysioNet EEGMMI: `https://physionet.org/content/eegmmidb/1.0.0/`
- Tsinghua SSVEP Benchmark: `https://bci.med.tsinghua.edu.cn/download.html`
- OpenNeuro ds003190: `https://openneuro.org/datasets/ds003190`
- Shin EEG-NIRS Figshare: `https://figshare.com/articles/dataset/9198932`
