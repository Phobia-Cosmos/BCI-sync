#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/progressive_feedback_proxy_runs/full49_odd25_subject18_seed4321}"
CLEAN_ROOT="${CLEAN_ROOT:-${REPO_ROOT}/experiments/canonical_n2n_shared_proxy/full49_task26_q20_l2p20_seed4321}"
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
  --progressive-proxy-tasks odd
  --progressive-clean-feedback-tasks even
  --progressive-base-subject 18
)

FEEDBACK=(
  --progressive-proxy-mode feedback
  --progressive-proxy-lr 1e-6
  --progressive-feedback-steps 4
  --progressive-feedback-batch 4
  --progressive-guide-epochs 2
  --progressive-generation-steps 3
  --progressive-generation-batch 4
  --progressive-reference-batch 4
  --progressive-step-relative-l2 0.01
  --progressive-step-linf-std 0.025
  --progressive-cumulative-relative-l2 0.20
  --progressive-cumulative-linf-std 0.50
  --progressive-history-weight 1.0
  --progressive-history-decay 0.8
  --progressive-input-cone-residual 0.5
  --progressive-source-weight 0.5
  --progressive-feedback-weight 1.0
  --progressive-feedback-decay 0.95
  --progressive-feedback-capacity 2500
)

run_stage() {
  local name="$1"
  local metrics="$2"
  shift 2
  local marker="${RUN_ROOT}/.${name}.complete"
  if [[ -f "${marker}" && -f "${metrics}" ]]; then
    printf 'skip completed stage: %s\n' "${name}"
    return
  fi
  "$@" 2>&1 | tee "${RUN_ROOT}/logs/${name}.log"
  test -f "${metrics}"
  touch "${marker}"
}

cd "${REPO_ROOT}"

run_stage \
  ewc_static \
  "${RUN_ROOT}/runs/ewc_static/ewc/metrics.json" \
  "${PYTHON}" experiments/regularization_cl_eeg.py \
    --methods ewc --ewc-strength 5000 --no-save-checkpoints \
    "${COMMON[@]}" \
    --progressive-proxy-mode static \
    --output-root "${RUN_ROOT}/runs/ewc_static"

run_stage \
  ewc_feedback \
  "${RUN_ROOT}/runs/ewc_feedback/ewc/metrics.json" \
  "${PYTHON}" experiments/regularization_cl_eeg.py \
    --methods ewc --ewc-strength 5000 --no-save-checkpoints \
    "${COMMON[@]}" "${FEEDBACK[@]}" \
    --output-root "${RUN_ROOT}/runs/ewc_feedback"

run_stage \
  plain_er_static \
  "${RUN_ROOT}/runs/plain_er_static/metrics.json" \
  "${PYTHON}" experiments/replay_cl_eeg.py \
    --method plain_er --memory-capacity 1000 --replay-ratio 1.0 \
    "${COMMON[@]}" \
    --progressive-proxy-mode static \
    --output-root "${RUN_ROOT}/runs/plain_er_static"

run_stage \
  plain_er_feedback \
  "${RUN_ROOT}/runs/plain_er_feedback/metrics.json" \
  "${PYTHON}" experiments/replay_cl_eeg.py \
    --method plain_er --memory-capacity 1000 --replay-ratio 1.0 \
    "${COMMON[@]}" "${FEEDBACK[@]}" \
    --output-root "${RUN_ROOT}/runs/plain_er_feedback"

"${PYTHON}" experiments/summarize_progressive_feedback_proxy.py \
  --run-root "${RUN_ROOT}" \
  --clean-root "${CLEAN_ROOT}" \
  --dose-results "${REPO_ROOT}/experiments/canonical_n2n_shared_proxy/dose_sweep_ewc_plain_er_nested_seed4321/FULL_RESULTS.json" \
  --bci "${BCI_PATH}" \
  | tee "${RUN_ROOT}/logs/summarize.log"

touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
