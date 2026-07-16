# Unlabeled SPR-EEG

This runner keeps the BrainUICL source-pretrained model and labeled source
memory, but never uses new-subject ground truth for training decisions.

For every new subject it performs:

1. CPC adaptation of the previous inference model.
2. Pseudo-label prediction for every sleep epoch.
3. Expert NT-Xent on all current sequences.
4. Base NT-Xent Self-Replay on current sequences plus P.
5. SCF grouping by pseudo-label, with no confidence gate.
6. Storage of accepted epoch references and their original pseudo-labels.
7. Masked supervised fine-tuning on P and old/new evaluation.

Confidence is recorded only as a diagnostic. Epochs below 0.9 are not removed.
New-subject labels loaded by the evaluation dataset are passed to the filter
only after selection to calculate pseudo-label error and P purity.

Base NT-Xent does not generate labels, examples, or a dataset. It updates the
long-lived feature extractor, Transformer and projection head so that two
augmentations of the same sequence have nearby representations while replay
reduces forgetting.

## Smoke test

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/spr_eeg_unlabeled/run.py \
  --output-root experiments/rttdp_brainuicl_runs/spr_unlabeled_smoke \
  --max-subjects 2 --guiding-epochs 1 --expert-epochs 1 \
  --base-epochs 1 --ft-epochs 1 --max-guiding-batches 1 \
  --max-ssl-batches 1 --max-ft-batches 1
```

## Ten-subject run

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/spr_eeg_unlabeled/run.py \
  --output-root experiments/rttdp_brainuicl_runs/spr_unlabeled_10sub_e10_seed4321 \
  --max-subjects 10 --guiding-epochs 10 --expert-epochs 10 \
  --base-epochs 10 --ft-epochs 10
```
