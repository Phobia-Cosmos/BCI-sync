# PACOL-style and BrainWash-style EEG poisoning comparison

## Scope and protocol

This is a small, single-seed attack validation after the clean regularization experiment. It is not a claim of exact reproduction of the image-based attacks.

- Victims: Finetune, SI, and MAS with the clean experiment's fixed hyperparameters and frozen student BatchNorm statistics.
- Stream: the first nine seed-4321 subjects. Only the final task, subject 80, is poisoned; its clean guiding-model pseudo-label ACC is about 0.735.
- Learner supervision: all guiding-model hard pseudo labels are accepted. There is no confidence filtering and no replay buffer.
- Injection: 3 of 43 current-task sequences are poisoned because a requested 5% rate is rounded up, giving an actual within-task rate of 6.98%.
- Budget: per-modality L-infinity perturbation at most `0.005 * std`, with five projected sign-gradient steps and a classifier-parameter attack surrogate.
- Attacker reference: source-training EEG inputs labeled by the current victim's hard predictions. These inputs are used only to construct attack gradients and never enter the learner optimizer.
- PACOL adaptation: match the poisoned current-data gradient to the gradient of pseudo-label-flipped historical proxy inputs.
- BrainWash adaptation: use the official method's one-step differentiable inner update. Reckless maximizes historical proxy loss; cautious also penalizes clean current-task loss.

The official BrainWash implementation at commit `660688c5eb9d9d8a58ba7728b1e98496ed292a9d` was inspected to verify the one-step update and outer-loss signs.

## Absolute results

| Setting | Victim | Final old ACC | Final old MF1 | Final seen ACC | Final seen MF1 | BWT ACC |
|---|---|---:|---:|---:|---:|---:|
| Clean | Finetune | 0.6829 | 0.6709 | 0.6276 | 0.5554 | -0.0083 |
| PACOL-style | Finetune | 0.6801 | 0.6670 | 0.6211 | 0.5484 | -0.0144 |
| BrainWash reckless | Finetune | 0.6799 | 0.6669 | 0.6211 | 0.5487 | -0.0143 |
| BrainWash cautious | Finetune | 0.6803 | 0.6672 | 0.6217 | 0.5490 | -0.0137 |
| Clean | SI | 0.7140 | 0.6965 | 0.6357 | 0.5483 | -0.0023 |
| PACOL-style | SI | 0.7142 | 0.6966 | 0.6351 | 0.5478 | -0.0029 |
| BrainWash reckless | SI | 0.7142 | 0.6966 | 0.6353 | 0.5480 | -0.0027 |
| BrainWash cautious | SI | 0.7141 | 0.6965 | 0.6353 | 0.5480 | -0.0027 |
| Clean | MAS | 0.7087 | 0.6879 | 0.6333 | 0.5474 | -0.0041 |
| PACOL-style | MAS | 0.7085 | 0.6877 | 0.6318 | 0.5458 | -0.0052 |
| BrainWash reckless | MAS | 0.7085 | 0.6877 | 0.6322 | 0.5460 | -0.0046 |
| BrainWash cautious | MAS | 0.7086 | 0.6878 | 0.6322 | 0.5460 | -0.0046 |

## Attack-induced ACC changes

| Victim | Attack | Delta old ACC | Delta final seen ACC | Delta BWT ACC |
|---|---|---:|---:|---:|
| Finetune | PACOL-style | -0.0029 | -0.0065 | -0.0061 |
| Finetune | BrainWash reckless | -0.0031 | -0.0065 | -0.0060 |
| Finetune | BrainWash cautious | -0.0026 | -0.0059 | -0.0054 |
| SI | PACOL-style | +0.0002 | -0.0006 | -0.0006 |
| SI | BrainWash reckless | +0.0002 | -0.0004 | -0.0004 |
| SI | BrainWash cautious | +0.0001 | -0.0004 | -0.0004 |
| MAS | PACOL-style | -0.0002 | -0.0015 | -0.0011 |
| MAS | BrainWash reckless | -0.0002 | -0.0010 | -0.0005 |
| MAS | BrainWash cautious | -0.0002 | -0.0010 | -0.0005 |

Tasks 1 through 8 have identical metric hashes across all four settings for every victim. The changes above therefore originate at the poisoned ninth task, not from earlier random trajectory differences.

## Optimization diagnostics

The attack objectives move in the intended direction. For Finetune, PACOL's gradient distance decreases from 0.7562 to 0.6533. BrainWash reckless's minimized negative old loss decreases from -0.10488 to -0.10498, meaning the surrogate old loss increases. Cautious BrainWash's combined objective decreases from 0.31104 to 0.31027.

The temporary clean guiding model retains 90.0% of epoch-level pseudo labels under PACOL, 95.0% under reckless BrainWash, and 96.7% under cautious BrainWash for Finetune; SI and MAS are similar. External labels would remain unchanged in the original clean-label papers, but pseudo labels are model outputs here and can change after input perturbation. This is a fundamental semantic difference in the unlabeled EEG setting.

## Stress check and conclusion

A separate Finetune sanity check raised the within-task injection rate to 20.9% and the budget to `0.01 * std`. The optimized surrogate objectives improved further, but final degradation did not increase monotonically: PACOL changed final seen ACC by -0.0062 and BrainWash by -0.0042 relative to Clean. The pseudo-label preservation rate also fell to 75%.

The defensible conclusion is narrow: under this low-budget, final-task, classifier-surrogate adaptation, both attacks produce small but measurable degradation for Finetune, while the fixed SI and MAS configurations largely absorb it. This does not establish general robustness to PACOL or BrainWash. The current BrainWash comparison uses direct source proxies rather than the paper's no-history model inversion, and both attacks optimize only the classifier surrogate instead of all victim parameters. Publication-level evaluation requires EEG model inversion or an explicitly declared direct-data upper bound, all-parameter or validated block-level unrolling, multiple subject orders, and an injection/budget sweep fixed independently of defense performance.
