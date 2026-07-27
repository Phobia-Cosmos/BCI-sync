#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/progressive_feedback_proxy_runs/position_sweep_full49_seed4321}"
ODD_ROOT="${ODD_ROOT:-${REPO_ROOT}/experiments/progressive_feedback_proxy_runs/cross_cl_v2_full49_seed4321}"
BCI_PATH="${BCI_PATH:-/home/undefined/Desktop/IPhone/BCI.md}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runs"

COMMON=(
  --seed 4321
  --batch 16
  --num-worker 0
  --ssl-epoch 10
  --incremental-epoch 10
  --ssl-lr 1e-6
  --cl-lr 1e-6
  --freeze-bn-stats
  --progressive-base-subject 18
  --progressive-upload-full-pool
)

V2=(
  --progressive-proxy-mode feedback
  --progressive-proxy-lr 1e-6
  --progressive-feedback-steps 4
  --progressive-feedback-batch 4
  --progressive-feedback-decay 0.95
  --progressive-feedback-capacity 2500
  --progressive-guide-epochs 1
  --progressive-generation-steps 20
  --progressive-generation-batch 4
  --progressive-reference-batch 16
  --progressive-step-relative-l2 0.05
  --progressive-step-linf-std 0.25
  --progressive-cumulative-relative-l2 0.20
  --progressive-cumulative-linf-std 0.50
  --progressive-fill-step-budget
  --progressive-input-cone-residual 0.2
  --progressive-history-weight 1.0
  --progressive-history-decay 0.8
  --progressive-history-refresh-count 16
  --progressive-source-weight 0.5
  --progressive-feedback-weight 1.0
  --progressive-target-weight 200
  --progressive-conflict-weight 20
  --progressive-gradient-norm-weight 1
  --progressive-virtual-old-weight 5
  --progressive-virtual-new-weight 5
  --progressive-confidence-weight 0
  --progressive-l2-weight 0
  --progressive-source-gate-samples 64
  --progressive-max-source-gradient-cosine 0.10
  --progressive-require-source-conflict
)

SCHEDULES=(
  "front_k05:1:5"
  "back_k05:45:49"
  "front_k10:1:10"
  "back_k10:40:49"
  "front_k25:1:25"
  "back_k25:25:49"
)

join_range() {
  local start="$1"
  local end="$2"
  seq -s, "${start}" "${end}"
}

complement_range() {
  local start="$1"
  local end="$2"
  local tasks=()
  local task
  for task in $(seq 1 49); do
    if (( task < start || task > end )); then
      tasks+=("${task}")
    fi
  done
  local IFS=,
  printf '%s' "${tasks[*]}"
}

run_stage() {
  local name="$1"
  local metrics="$2"
  shift 2
  local marker="${RUN_ROOT}/.${name}.complete"
  local log="${RUN_ROOT}/logs/${name}.log"
  if [[ -f "${marker}" && -f "${metrics}" ]]; then
    printf 'skip completed stage: %s\n' "${name}"
    return
  fi
  local started="${SECONDS}"
  printf 'start stage: %s\n' "${name}"
  if "$@" >"${log}" 2>&1; then
    test -f "${metrics}"
    touch "${marker}"
    printf 'complete stage: %s (%ss)\n' "${name}" "$((SECONDS - started))"
  else
    tail -80 "${log}" >&2
    return 1
  fi
}

cd "${REPO_ROOT}"

for schedule in "${SCHEDULES[@]}"; do
  IFS=: read -r name start end <<<"${schedule}"
  proxy_tasks="$(join_range "${start}" "${end}")"
  clean_tasks="$(complement_range "${start}" "${end}")"
  schedule_args=(
    --progressive-proxy-tasks "${proxy_tasks}"
    --progressive-clean-feedback-tasks "${clean_tasks}"
  )

  run_stage \
    "ewc_${name}_static48" \
    "${RUN_ROOT}/runs/ewc_${name}_static48/ewc/metrics.json" \
    "${PYTHON}" experiments/regularization_cl_eeg.py \
      --methods ewc --ewc-strength 5000 --no-save-checkpoints \
      "${COMMON[@]}" "${schedule_args[@]}" \
      --progressive-proxy-mode static \
      --output-root "${RUN_ROOT}/runs/ewc_${name}_static48"

  run_stage \
    "ewc_${name}_v2" \
    "${RUN_ROOT}/runs/ewc_${name}_v2/ewc/metrics.json" \
    "${PYTHON}" experiments/regularization_cl_eeg.py \
      --methods ewc --ewc-strength 5000 --no-save-checkpoints \
      "${COMMON[@]}" "${schedule_args[@]}" "${V2[@]}" \
      --output-root "${RUN_ROOT}/runs/ewc_${name}_v2"

  run_stage \
    "plain_er_${name}_static48" \
    "${RUN_ROOT}/runs/plain_er_${name}_static48/metrics.json" \
    "${PYTHON}" experiments/replay_cl_eeg.py \
      --method plain_er --memory-capacity 1000 --replay-ratio 1.0 \
      "${COMMON[@]}" "${schedule_args[@]}" \
      --progressive-proxy-mode static \
      --output-root "${RUN_ROOT}/runs/plain_er_${name}_static48"

  run_stage \
    "plain_er_${name}_v2" \
    "${RUN_ROOT}/runs/plain_er_${name}_v2/metrics.json" \
    "${PYTHON}" experiments/replay_cl_eeg.py \
      --method plain_er --memory-capacity 1000 --replay-ratio 1.0 \
      "${COMMON[@]}" "${schedule_args[@]}" "${V2[@]}" \
      --output-root "${RUN_ROOT}/runs/plain_er_${name}_v2"
done

"${PYTHON}" experiments/summarize_proxy_position_sweep.py \
  --run-root "${RUN_ROOT}" \
  --odd-root "${ODD_ROOT}" \
  --bci "${BCI_PATH}" \
  >"${RUN_ROOT}/logs/summarize.log"

touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
printf 'position sweep complete: %s\n' "${RUN_ROOT}"
