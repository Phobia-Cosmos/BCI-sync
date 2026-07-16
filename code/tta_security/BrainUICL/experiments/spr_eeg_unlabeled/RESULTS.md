# Unlabeled SPR-EEG Results

## Configuration

- Seed: 4321
- New subjects: first 10 BrainUICL new subjects
- Source model: BrainUICL source-pretrained checkpoint
- Guiding model: previous inference model adapted with CPC
- Guiding/expert/base/fine-tune epochs: 10/10/10/10
- Confidence gate: disabled
- Candidate epochs: every epoch from every current-subject sequence
- P: 3000 labeled source epochs + 2000 dynamic pseudo-labeled epochs

Local metrics:

```text
experiments/rttdp_brainuicl_runs/spr_unlabeled_10sub_e10_seed4321/metrics.json
```

## Results

| Method | Old ACC | Old MF1 | AAA | AAF1 | FR | New after ACC | New after MF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL, 10 epochs | 0.6676 | 0.6247 | 0.7076 | 0.6837 | 0.0496 | 0.5946 | 0.5235 |
| Unlabeled SPR-EEG | **0.6898** | **0.6615** | 0.6996 | 0.6724 | **0.0180** | **0.6195** | **0.5471** |
| Oracle-label SPR-EEG | 0.7123 | 0.6881 | 0.7153 | 0.6954 | 0.0140 | 0.6424 | 0.5746 |

Unlabeled SPR improves final old ACC/MF1 over BrainUICL by 2.22/3.68 points
and new after ACC/MF1 by 2.48/2.36 points. Its AAA/AAF1 are 0.80/1.12 points
lower, so it does not dominate BrainUICL throughout the trajectory. Relative
to oracle-label SPR, pseudo-labeling costs about 2.25 old ACC, 2.67 old MF1,
2.29 new ACC and 2.75 new MF1 points.

## Pseudo-Label Filtering

| Diagnostic | Value |
| --- | ---: |
| Candidate epochs | 8880 |
| Epochs with confidence below 0.9 | 2511 |
| Accepted epochs | 5451 |
| Acceptance rate | 61.39% |
| Pseudo-label error before SCF | 33.07% |
| Accepted-set error after SCF | 27.92% |
| Final dynamic P purity | 76.20% |
| Final total P purity | 90.48% |

The filter reduces pseudo-label error without a confidence gate, but coherent
errors remain. Subject 26 has 62.02% pseudo-label error before SCF and 59.93%
after SCF, demonstrating that graph centrality cannot repair a large coherent
wrong cluster.

New-subject ground truth is used only after selection for metrics and record
purity diagnostics. Selection depends only on pseudo-labels, expert embeddings
and seeded randomness; tests verify that changing diagnostic labels cannot
change accepted epochs.

## Full 49-Subject Results

The full run keeps training batch 8 and increases only evaluation batch to 32.
It completes all 49 new subjects in about 13.8 minutes on an RTX 4070 SUPER.

```text
experiments/rttdp_brainuicl_runs/spr_unlabeled_full49_e10_seed4321/
```

| Method | Old ACC | Old MF1 | AAA | AAF1 | Last-10 ACC | New after ACC | New after MF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL | 0.6569 | 0.6231 | 0.6934 | 0.6685 | 0.6969 | 0.6182 | 0.5548 |
| Unlabeled SPR-EEG | **0.7109** | **0.6815** | **0.7037** | **0.6766** | **0.7066** | **0.6694** | **0.5951** |
| Oracle-label SPR-EEG | 0.7126 | 0.6923 | 0.7147 | 0.6933 | 0.7210 | 0.6864 | 0.6167 |

Relative to BrainUICL, unlabeled SPR improves final old ACC/MF1 by 5.40/5.84
points and new after ACC/MF1 by 5.12/4.03 points. The subject-paired new-after
differences are +5.12 ACC points (95% bootstrap CI 2.56 to 7.95,
Wilcoxon p=0.0013) and +4.03 MF1 points (CI 1.96 to 6.17, p=0.0016), with
34/49 subject wins for both metrics.

Relative to oracle SPR, unlabeled pseudo-labeling costs 0.17 old ACC, 1.08 old
MF1, 1.70 new ACC and 2.16 new MF1 points. Oracle SPR retains a stronger
trajectory, with AAA/AAF1 higher by 1.10/1.68 points.

### Full-Stream Pseudo-Label Diagnostics

| Diagnostic | Value |
| --- | ---: |
| Candidate epochs | 42960 |
| Confidence below 0.9 but retained as candidates | 12243 |
| Accepted epochs | 26934 |
| Acceptance rate | 62.70% |
| Weighted pseudo-label error before SCF | 30.09% |
| Weighted accepted-set error | 26.35% |
| Subjects with reduced error | 45/49 |
| Final dynamic P purity | 77.95% |
| Final total P purity | 91.18% |

SCF reduces weighted pseudo-label error by 3.74 points without a confidence
gate. It fails on four subjects (26, 98, 2 and 15), where accepted-set error is
higher, confirming that centrality can preserve coherent wrong clusters.

## Extra 40% Pseudo-Label Noise

An additional full 49-subject run corrupts each guiding pseudo-label with 40%
probability after prediction. This is a stress test on top of natural
pseudo-label error, not a stream with exactly 40% total error.

| Variant | Old ACC | Old MF1 | AAA | Last-10 ACC | New after ACC | New after MF1 | Dynamic P purity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Natural pseudo-labels | 0.7109 | 0.6815 | 0.7037 | 0.7066 | 0.6694 | 0.5951 | 77.95% |
| +40% pseudo noise | 0.6953 | 0.6694 | 0.7018 | 0.7016 | 0.6405 | 0.5767 | 63.90% |

The extra-noise run adds 17,251 corruptions across 42,960 epochs. Effective
pseudo-label error is 55.49% before SCF and 41.05% in the accepted set. SCF
therefore removes substantial independent random noise, but dynamic P purity
drops by 14.05 points and new after ACC/MF1 drop by 2.89/1.83 points.

## Unified Variant Matrix

| Variant | Label source | Old ACC | Old MF1 | AAA | New after ACC | New after MF1 | Dynamic P purity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL | pseudo-label + confidence gate | 0.6569 | 0.6231 | 0.6934 | 0.6182 | 0.5548 | sequence buffer |
| Oracle SPR | observed clean label | 0.7126 | 0.6923 | 0.7147 | 0.6864 | 0.6167 | 100% |
| Unlabeled SPR | natural pseudo-label | 0.7109 | 0.6815 | 0.7037 | 0.6694 | 0.5951 | 77.95% |
| Unlabeled SPR + noise | pseudo-label + extra 40% | 0.6953 | 0.6694 | 0.7018 | 0.6405 | 0.5767 | 63.90% |
| Random-init SPR clean | observed clean label | 0.6663 | 0.6291 | 0.7061 | 0.6804 | 0.5441 | 100% |
| Random-init SPR 40% | observed label + 40% noise | 0.7201 | 0.6754 | 0.6539 | 0.6176 | 0.4623 | 61.60% |

Random-init new metrics use held-out sequences, whereas BrainUICL, oracle and
unlabeled SPR use the original transductive full-subject protocol. Those rows
must not be interpreted as a controlled ranking.

