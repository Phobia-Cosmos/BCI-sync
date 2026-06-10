# Directory Organization and CPGA Toy Run

Date: 2026-06-10

## Organization

The repository was reorganized into content-oriented directories:

- `papers/sfda/`
- `papers/bci_security/`
- `papers/bci_decoding/`
- `papers/accelerator/`
- `papers/bci_realtime/`
- `code/sfda/`
- `code/brain_decoding/`
- `code/bci_toolkits/`
- `reports/`
- `scripts/`

The root directory now keeps only repository-level files and high-level folders.
The local `data/` directory remains ignored by Git.

## Code Layout

The code directory now uses one paper/project per subdirectory:

- `code/sfda/2020ICML-SHOT`
- `code/sfda/2021Arxiv-CPGA-Avatar-Prototype-Generation-Adaptation`
- `code/sfda/2023Arxiv-SFDA-SSVEP-BCI`
- `code/sfda/2024CVPR-DIFO-Plus`
- `code/sfda/2024CVPR-DIFO-ProDe-Source-Free-Domain-Adaptation`
- `code/sfda/2025AAAI-SF-UIDA`
- `code/sfda/2025JBHI-PDCC-Cross-Subject-EEG-Classification`
- `code/brain_decoding/2022NeurIPS-TSMNet-SPD-DSBN-EEG-UDA`
- `code/brain_decoding/2024CVPR-MindBridge`
- `code/bci_toolkits/Deep-BCI`

## CPGA Code Status

No official public CPGA GitHub repository was found during search. The created
CPGA directory therefore contains a toy privacy probe, not an official CPGA
implementation.

Script:

```bash
python3 -B code/sfda/2021Arxiv-CPGA-Avatar-Prototype-Generation-Adaptation/cpga_toy_privacy_probe.py
```

Observed output:

```text
Source-model confidence for generated avatars:
[[0.9817 0.0088 0.0095]
 [0.0091 0.983  0.0079]
 [0.0102 0.0082 0.9817]]

Nearest hidden center for each avatar: [0, 1, 2]
Mean nearest-center distance: 1.5631
```

Interpretation:

The generated avatar prototypes were produced using only the frozen source
model, after discarding source data. They still recovered class-level source
prototypes. This supports the security concern that source models used in SFDA
can leak source distribution information even if raw source data is unavailable.

