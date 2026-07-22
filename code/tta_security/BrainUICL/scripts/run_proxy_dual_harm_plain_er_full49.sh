#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-experiments/replay_cl_eeg_runs/proxy_dual_harm_plain_er_full49}"
ATTACK_TASKS="1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49"

COMMON_ARGS=(
  --gpu 0
  --max-subjects 0
  --ssl-epoch 10
  --incremental-epoch 10
  --batch 16
  --num-worker 0
  --freeze-bn-stats
  --memory-capacity 1000
  --replay-ratio 1.0
  --retention-milestones 10,25,49
)

summary_complete() {
  local path="$1"
  [[ -f "$path" ]] && jq -e 'has("plain_er")' "$path" >/dev/null
}

run_condition() {
  local condition="$1"
  shift
  local output_root="$RUN_ROOT/$condition"
  local log_path="$RUN_ROOT/$condition.log"
  if summary_complete "$output_root/summary.json"; then
    printf '[skip] %s is complete\n' "$condition"
    return
  fi
  printf '[run] %s\n' "$condition"
  "$PYTHON" experiments/replay_cl_eeg.py \
    --output-root "$output_root" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    > "$log_path" 2>&1
}

mkdir -p "$RUN_ROOT"

run_condition clean

run_condition repeat_clean \
  --attack-mode benign_repeat \
  --attack-tasks "$ATTACK_TASKS" \
  --attack-proxy-repeat 3

run_condition attack_shared \
  --attack-mode proxy_dual_harm \
  --attack-tasks "$ATTACK_TASKS" \
  --attack-fraction 1.0 \
  --attack-eps-scale 0.50 \
  --attack-max-relative-l2 0.20 \
  --attack-steps 3 \
  --attack-reference-batch 4 \
  --attack-generation-batch 4 \
  --attack-random-start \
  --attack-target-weight 5.0 \
  --attack-conflict-weight 1.0 \
  --attack-gradient-norm-weight 0.25 \
  --attack-virtual-old-weight 1.0 \
  --attack-virtual-new-weight 1.0 \
  --attack-new-proxy-weight 1.0 \
  --attack-curvature-scale 0.0 \
  --attack-min-confidence 0.85 \
  --attack-confidence-weight 2.0 \
  --attack-l2-weight 0.01 \
  --attack-proxy-repeat 3

"$PYTHON" experiments/summarize_proxy_dual_harm_replay.py \
  --run-root "$RUN_ROOT" \
  > "$RUN_ROOT/summary.log" 2>&1

"$PYTHON" -m unittest \
  tests.test_replay_cl_eeg \
  tests.test_regularization_cl_eeg \
  tests.test_summarize_proxy_dual_harm \
  > "$RUN_ROOT/tests.log" 2>&1

touch "$RUN_ROOT/_EXECUTION_COMPLETE"
printf '[complete] plain ER dual-harm experiment finished\n'
