#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
STREAM_ROOT="${STREAM_ROOT:-experiments/frozen_proxy_frequency_shift/full49_seed4321}"
RUN_ROOT="${RUN_ROOT:-experiments/frozen_proxy_frequency_shift/full49_runs}"
METHODS="ewc,online_ewc,si,mas"
STREAMS=(clean I-NS I-S F-NS F-S)

COMMON_ARGS=(
  --methods "$METHODS"
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
)

mkdir -p "$RUN_ROOT"

summary_complete() {
  local summary_path="$1"
  [[ -f "$summary_path" ]] && \
    jq -e '(["ewc", "online_ewc", "si", "mas"] - keys) | length == 0' \
      "$summary_path" >/dev/null
}

ensure_streams() {
  if [[ -f "$STREAM_ROOT/manifest.json" ]] && \
     jq -e '.proxy_parameters_unchanged == true' "$STREAM_ROOT/manifest.json" >/dev/null
  then
    return
  fi

  "$PYTHON" experiments/generate_frozen_proxy_frequency_shift.py \
    --output-root "$STREAM_ROOT" \
    --max-subjects 49 \
    --frequent-count 25 \
    --infrequent-count 3 \
    --sequence-fraction 0.20 \
    --relative-l2-budgets 0.05 \
    --linf-std-scale 0.20 \
    --direction-steps 5 \
    --direction-batch 3 \
    --materialize-batch 12 \
    > "$RUN_ROOT/generation.log" 2>&1
}

run_condition() {
  local defense="$1"
  local condition="$2"
  shift 2
  local output_root="$RUN_ROOT/$defense/$condition"
  local log_path="$RUN_ROOT/$defense/${condition}.log"

  mkdir -p "$RUN_ROOT/$defense"
  if summary_complete "$output_root/summary.json"
  then
    printf '[skip] %s/%s is complete\n' "$defense" "$condition"
    return
  fi

  printf '[run] %s/%s\n' "$defense" "$condition"
  "$PYTHON" experiments/regularization_cl_eeg.py \
    --output-root "$output_root" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    > "$log_path" 2>&1
}

run_matrix() {
  local defense="$1"
  shift
  local defense_args=("$@")
  local condition

  for condition in "${STREAMS[@]}"
  do
    if [[ "$condition" == "clean" ]]
    then
      run_condition "$defense" "$condition" "${defense_args[@]}"
    else
      run_condition "$defense" "$condition" \
        "${defense_args[@]}" \
        --noise-upload-root "$STREAM_ROOT/rel_l2_0500/$condition"
    fi
  done
}

ensure_streams

run_matrix none --defense-mode none

run_matrix robust_feature \
  --defense-mode robust_feature \
  --robust-feature-budget-per-dimension 0.026041666666666668

run_condition t2t clean \
  --defense-mode t2t \
  --t2t-action rollback \
  --t2t-param-scope all

run_condition t2t I-S \
  --defense-mode t2t \
  --t2t-action rollback \
  --t2t-param-scope all \
  --noise-upload-root "$STREAM_ROOT/rel_l2_0500/I-S"

"$PYTHON" -m unittest \
  tests.test_frozen_proxy_frequency_shift \
  tests.test_icml2026_cl_defenses \
  tests.test_regularization_cl_eeg \
  > "$RUN_ROOT/final_tests.log" 2>&1

touch "$RUN_ROOT/_EXECUTION_COMPLETE"
printf '[complete] all scheduled full49 regularization experiments finished\n'
