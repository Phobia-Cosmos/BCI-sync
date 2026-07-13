# EEG Experiment Workspace

The EEG experiments are centered in `papers/TTAP/BrainUICL/experiments/`.
Experiment runners stay separate from generated artifacts, and the original
BrainUICL training entry point remains `papers/TTAP/BrainUICL/main.py`.

## Runners

- `rttdp_brainuicl_full.py`: shared full BrainUICL attack/defense runner.
- `puridiver_eeg.py`: PuriDivER defense experiments built on BrainUICL.
- `pure_puridiver_eeg.py`: pure labeled-stream PuriDivER/ER comparisons.
- `unlabeled_puridiver_eeg.py`: unlabeled pseudo-label PuriDivER comparisons.
- `spr_eeg_pure.py`: labeled SPR-EEG experiments.
- `spr_eeg_random_init/`: random-initialized SPR protocols.
- `spr_eeg_unlabeled/`: source-pretrained unlabeled SPR protocol.
- `attack_diagnostics.py`, `distribution_trajectory.py`,
  `stable_neuron_analysis.py`, and `proxy_degradation_probe.py`: analyses and
  diagnostics for completed runs.
- `finalize_attack_only.py`: post-processes attack-only runs into comparison
  artifacts.

## Generated Artifacts

- `rttdp_brainuicl_runs/`: per-run metrics, splits, logs, pseudo-labels, and
  checkpoints.
- `attack_diagnostics/`, `distribution_trajectory/`,
  `stable_neuron_analysis/`, and `proxy_degradation/`: derived analyses.
- `code_checkpoints/`: point-in-time source snapshots used to document a run.

Large generated directories, model weights, EEG arrays, caches, and logs are
local-only and ignored by Git. Keep durable conclusions in the Markdown
reports at the BrainUICL root or in protocol-specific `RESULTS.md` files.

## Run Naming and Retention

Use a new output directory for every run and include the scope, method,
condition, and seed in its name. Existing names use these prefixes:

- `smoke*`: minimal end-to-end validation.
- `probe*`: short diagnostic or parameter probe.
- `full49*`: full 49-subject evaluation.

For a result that will be cited, retain its configuration/split,
`metrics.json`, and a concise Markdown summary. Treat `pid.txt` as a launch
record only; verify the process is still running before relying on it. Remove
or archive smoke and probe artifacts only after their useful conclusions have
been copied into a durable report.
