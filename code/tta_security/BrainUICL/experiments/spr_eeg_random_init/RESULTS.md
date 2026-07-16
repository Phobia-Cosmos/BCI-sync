# Random-Initialized SPR-EEG Results

## Configuration

- Seed: 4321
- Initialization: random base, random per-chunk expert
- Initial Purified Buffer: empty
- Memory capacity: 5000 epoch references
- Labels: observed ISRUC labels, no artificial noise in this run
- Sequence split: 80% adaptation, 20% held-out evaluation
- Expert/base/fine-tune: 10/10/10 epochs
- Progression subjects: 10
- Fixed-split warmup subjects: all 24 BrainUICL train subjects
- Fixed-split new subjects: first 10 BrainUICL new subjects

Local metrics are under:

```text
experiments/rttdp_brainuicl_runs/spr_random_init_10sub_e10_seed4321/
```

The directory is excluded by `.gitignore`.

## Task Progression Protocol

| Metric | ACC | MF1 |
| --- | ---: | ---: |
| Random initial model | 0.1270 | 0.0419 |
| Before current-subject adaptation | 0.5406 | 0.4220 |
| After current-subject adaptation | 0.6679 | 0.5363 |

After subject 10, the three earliest anchor subjects had ACC values 0.6944,
0.8375 and 0.6500. P grew from empty to 3968 retained epochs.

## Fixed Subject Split Protocol

| Stage | Old ACC | Old MF1 |
| --- | ---: | ---: |
| Random model before warmup | 0.1117 | 0.0402 |
| After 24 source subjects as SPR stream | 0.6545 | 0.6239 |
| After 10 sequential new subjects | 0.6868 | 0.6653 |

The final trajectory metrics were AAA 0.6942, AAF1 0.6703 and FR 0.0493.
New-subject held-out ACC improved from 0.5832 before adaptation to 0.6467
after adaptation; held-out MF1 improved from 0.4583 to 0.5423. P reached its
5000-epoch capacity without a protected source partition.

## Interpretation

Random initialization is initially unusable for sleep staging. Processing the
24 source subjects through the same SPR path, instead of supervised
pretraining or direct source-memory insertion, recovers most of the old-subject
performance. The final old ACC remains below the source-pretrained 10-subject
SPR result (0.6868 versus 0.7123), although the evaluation splits differ:
random-init new-subject metrics use held-out sequences, while the earlier
oracle runner evaluated the same subject sequences used during adaptation.

This run uses clean observed labels. A random classifier cannot bootstrap
meaningful sleep-stage pseudo-labels, so fully unlabeled random initialization
requires an additional clustering or warm-start mechanism and is not evaluated
here.

## Full Individual Results

The full run uses all 98 subjects in task progression. The fixed-split run
uses all 24 source subjects as SPR warmup chunks and all 49 new subjects, while
the 19 old subjects remain evaluation-only. Clean and 40% symmetric noise both
completed with the same 10/10/10 epoch budget.

```text
experiments/rttdp_brainuicl_runs/spr_random_init_full_e10_seed4321/
```

### Progression Across All 98 Subjects

| Noise | Before ACC | After ACC | Before MF1 | After MF1 | Final P purity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.6438 | 0.6909 | 0.5256 | 0.5593 | 100% |
| 40% | 0.6110 | 0.6295 | 0.4832 | 0.4921 | 61.34% |

For clean data, current-subject adaptation improves ACC by 4.71 points and
MF1 by 3.37 points. Under 40% noise, the gains shrink to 1.85 and 0.89 points.

At the end of 98 subjects, the clean ACC values of the first three anchors
are 0.6000, 0.7500 and 0.7056. Their mean final ACC is 0.6852 versus 0.6146
immediately after first learning, but 10.97 points below their mean historical
peak. Under 40% noise, final anchor ACC averages 0.6507 and is 13.56 points
below the mean peak.

### Full Fixed-Split Comparison With BrainUICL

| Method | Noise | Final old ACC | Old MF1 | AAA | AAF1 | Last-10 old ACC | New after ACC | New after MF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL | 0% | 0.6569 | 0.6231 | 0.6934 | 0.6685 | 0.6969 | 0.6182 | 0.5548 |
| Random-init SPR | 0% | 0.6663 | 0.6291 | 0.7061 | 0.6837 | 0.6971 | 0.6804 | 0.5441 |
| BrainUICL | 40% | 0.6650 | 0.6249 | 0.6876 | 0.6614 | 0.6891 | 0.6104 | 0.5517 |
| Random-init SPR | 40% | 0.7201 | 0.6754 | 0.6539 | 0.6208 | 0.6622 | 0.6176 | 0.4623 |

The clean endpoint favors random-init SPR by 0.95 ACC and 0.60 MF1 points,
but the last-10 old ACC values are effectively equal. Under 40% noise, the
random-init endpoint is 5.51 ACC points higher, yet its AAA is 3.37 points
lower and last-10 old ACC is 2.69 points lower. The final endpoint is therefore
not representative of the noisy trajectory.

Random-init SPR uses observed labels and held-out new-subject sequences;
BrainUICL uses teacher pseudo-labels and evaluates all current-subject
sequences. The table is a protocol-level comparison, not a controlled claim
that random-init SPR is superior under identical supervision.

### Filtering Conclusion

The 40% fixed stream contains 20,282 corrupted epochs. SCF rejects 9,049 of
them (44.62%), but the final P purity is only 61.60%, barely above the roughly
60% clean proportion in the input. By contrast, source-pretrained SPR reaches
85.45% dynamic P purity in the prior experiment. Random expert features are
therefore inadequate for strong early purification under this 10-epoch budget.
