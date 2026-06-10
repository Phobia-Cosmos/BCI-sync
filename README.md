# BCI-sync

BCI/EEG research workspace synchronized for reading papers, running source-free
domain adaptation code, and tracking security/hardware notes.

## Layout

- `papers/sfda/`: source-free domain adaptation and cross-subject adaptation papers.
- `papers/bci_security/`: BCI/EEG privacy, attack, defense, and security survey papers.
- `papers/bci_decoding/`: neural decoding, EEG foundation models, reconstruction, and real-time decoding papers.
- `papers/accelerator/`: BCI system architecture, DNN accelerator, low-power accelerator papers, and accelerator notes.
- `papers/bci_realtime/`: real-time BCI platform and pipeline papers.
- `code/sfda/`: one subdirectory per SFDA-related paper or codebase.
- `code/brain_decoding/`: one subdirectory per brain decoding codebase.
- `code/bci_toolkits/`: broad BCI toolkits or multi-paper code collections.
- `reports/`: local survey notes, run logs, and research planning documents.
- `scripts/`: local utility scripts.
- `data/`: local datasets only; ignored by Git and not synchronized.

## Code Sync

Most external codebases are registered as Git submodules. After cloning this
repository on another machine, run:

```bash
git submodule update --init --recursive
```

## CPGA Note

No official public CPGA GitHub repository was found for
`Source-free Domain Adaptation via Avatar Prototype Generation and Adaptation`.
The directory `code/sfda/2021Arxiv-CPGA-Avatar-Prototype-Generation-Adaptation/`
contains a small NumPy toy probe that demonstrates the privacy intuition:
a frozen source model can be used to generate source-like avatar prototypes even
when source data is unavailable.
