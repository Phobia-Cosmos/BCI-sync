# Random-Initialized SPR-EEG

This directory contains an SPR experiment that does not load the BrainUICL
source-pretrained checkpoint and does not preload source epochs into the
Purified Buffer. Base and expert networks start randomly, `D=P=empty`, and
every retained epoch must pass the same Self-Centered Filter.

## Protocols

`progression` treats a shuffled subject stream as sequential tasks. Each
subject is split into adaptation and held-out sequences. After every subject,
the runner evaluates the current subject and the earliest anchor subjects.

`fixed_split` retains BrainUICL's train/val/old/new subject split. The train
subjects are not used for supervised pretraining; they arrive sequentially as
ordinary SPR warmup chunks. The fixed old subjects never enter training. Each
new subject is split into adaptation and held-out sequences and is evaluated
after adaptation.

Both protocols use observed ISRUC sleep-stage labels, optionally corrupted by
symmetric noise, because original SPR requires an observed label for each
sample. A random classifier cannot provide meaningful pseudo-labels. This is a
labeled noisy-stream experiment, not an unsupervised deployment protocol.

## Smoke test

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/spr_eeg_random_init/run.py \
  --protocol both --max-subjects 2 --max-warmup-subjects 2 \
  --noise-rates 0.0 --expert-epochs 1 --base-epochs 1 --ft-epochs 1 \
  --max-ssl-batches 1 --max-ft-batches 1
```

## Full run

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/spr_eeg_random_init/run.py \
  --protocol both --max-subjects 0 --max-warmup-subjects 0 \
  --noise-rates 0.0 0.4 --expert-epochs 10 --base-epochs 10 --ft-epochs 10
```
