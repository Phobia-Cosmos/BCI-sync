#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-experiments/replay_cl_eeg_runs/fixed_shared_defenses_full49}"
FIXED_UPLOAD_ROOT="${FIXED_UPLOAD_ROOT:-experiments/replay_cl_eeg_runs/proxy_dual_harm_plain_er_full49/attack_shared}"
ATTACK_TASKS="1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49"

if [[ ! -f "$FIXED_UPLOAD_ROOT/metrics.json" ]]; then
  printf 'Missing fixed-upload metrics: %s\n' "$FIXED_UPLOAD_ROOT/metrics.json" >&2
  exit 1
fi

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
  local method="$1"
  local path="$2"
  [[ -f "$path" ]] && jq -e --arg method "$method" 'has($method)' "$path" >/dev/null
}

run_condition() {
  local method="$1"
  local condition="$2"
  shift 2
  local output_root="$RUN_ROOT/$method/$condition"
  local log_path="$RUN_ROOT/$method/$condition.log"
  if summary_complete "$method" "$output_root/summary.json"; then
    printf '[skip] %s/%s is complete\n' "$method" "$condition"
    return
  fi
  mkdir -p "$(dirname "$log_path")"
  printf '[run] %s/%s\n' "$method" "$condition"
  "$PYTHON" experiments/replay_cl_eeg.py \
    --method "$method" \
    --output-root "$output_root" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    > "$log_path" 2>&1
}

mkdir -p "$RUN_ROOT"

for method in plain_er spr_er puridiver_memory_ce puridiver_cru; do
  run_condition "$method" clean
  run_condition "$method" repeat_clean \
    --attack-mode benign_repeat \
    --attack-tasks "$ATTACK_TASKS" \
    --attack-proxy-repeat 3
  run_condition "$method" attack_fixed \
    --fixed-upload-root "$FIXED_UPLOAD_ROOT" \
    --attack-mode proxy_dual_harm \
    --attack-tasks "$ATTACK_TASKS" \
    --attack-proxy-repeat 3
done

"$PYTHON" experiments/summarize_fixed_shared_replay_defenses.py \
  --run-root "$RUN_ROOT" \
  > "$RUN_ROOT/summary.log" 2>&1

"$PYTHON" -m unittest \
  tests.test_replay_cl_eeg \
  tests.test_aligned_replay_defenses \
  tests.test_summarize_proxy_dual_harm \
  > "$RUN_ROOT/tests.log" 2>&1

touch "$RUN_ROOT/_EXECUTION_COMPLETE"
printf '[complete] fixed shared replay defense matrix finished\n'
