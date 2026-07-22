#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
STREAM_ROOT="${STREAM_ROOT:-experiments/ewc_attack_strength_sweep/frozen_proxy_F-S}"
RUN_ROOT="${RUN_ROOT:-experiments/ewc_attack_strength_sweep/runs_ewc}"
CLEAN_REFERENCE="${CLEAN_REFERENCE:-experiments/ewc_attack_strength_sweep/runs_ewc/clean_repeat}"

COMMON_ARGS=(
  --methods ewc
  --max-subjects 49
  --ssl-epoch 10
  --incremental-epoch 10
  --freeze-bn-stats
  --no-save-checkpoints
  --retention-milestones 10,25,49
  --ewc-strength 5000
  --online-ewc-strength 6500
  --online-ewc-decay 1
  --si-strength 1500000
  --si-xi 0.000001
  --mas-strength 3000
  --mas-decay 1
  --defense-mode none
)

summary_complete() {
  local summary_path="$1"
  [[ -f "$summary_path" ]] && \
    jq -e '(["ewc"] - keys) | length == 0' "$summary_path" >/dev/null
}

mkdir -p "$RUN_ROOT"

if [[ ! -f "$STREAM_ROOT/manifest.json" ]]; then
  "$PYTHON" experiments/generate_ewc_attack_strength_sweep.py \
    --output-root "$STREAM_ROOT" \
    > "$RUN_ROOT/generation.log" 2>&1
fi

if ! summary_complete "$CLEAN_REFERENCE/summary.json"; then
  printf '[run] clean reference\n'
  mkdir -p "$CLEAN_REFERENCE"
  "$PYTHON" experiments/regularization_cl_eeg.py \
    --output-root "$CLEAN_REFERENCE" \
    "${COMMON_ARGS[@]}" \
    > "$RUN_ROOT/clean_repeat.log" 2>&1
fi

cp "$CLEAN_REFERENCE/summary.json" "$RUN_ROOT/clean_reference_summary.json"
cp "$CLEAN_REFERENCE/ewc/metrics.json" "$RUN_ROOT/clean_reference_metrics.json"

mapfile -t CONDITIONS < <(
  jq -r '.conditions[].condition' "$STREAM_ROOT/manifest.json"
)

for condition in "${CONDITIONS[@]}"; do
  output_root="$RUN_ROOT/$condition"
  log_path="$RUN_ROOT/$condition.log"
  if summary_complete "$output_root/summary.json"; then
    printf '[skip] %s is complete\n' "$condition"
    continue
  fi
  printf '[run] %s\n' "$condition"
  "$PYTHON" experiments/regularization_cl_eeg.py \
    --output-root "$output_root" \
    "${COMMON_ARGS[@]}" \
    --noise-upload-root "$STREAM_ROOT/$condition" \
    > "$log_path" 2>&1
done

"$PYTHON" experiments/summarize_ewc_attack_strength_sweep.py \
  --run-root "$RUN_ROOT" \
  --stream-root "$STREAM_ROOT" \
  --clean-reference "$CLEAN_REFERENCE" \
  > "$RUN_ROOT/summary.log" 2>&1

touch "$RUN_ROOT/_EXECUTION_COMPLETE"
printf '[complete] EWC attack-strength sweep finished\n'
