#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/progressive_feedback_proxy_runs/cross_cl_v2_full49_seed4321}"
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

V2_STRONG=(
  "${V2[@]}"
  --progressive-step-relative-l2 0.10
  --progressive-step-linf-std 0.50
  --progressive-cumulative-relative-l2 0.40
  --progressive-cumulative-linf-std 1.00
)

V2_STRONG_C60=(
  "${V2[@]}"
  --progressive-step-relative-l2 0.15
  --progressive-step-linf-std 0.75
  --progressive-cumulative-relative-l2 0.60
  --progressive-cumulative-linf-std 1.50
  --progressive-max-source-gradient-cosine 0.20
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
  ewc_static48 \
  "${RUN_ROOT}/runs/ewc_static48/ewc/metrics.json" \
  "${PYTHON}" experiments/regularization_cl_eeg.py \
    --methods ewc --ewc-strength 5000 --no-save-checkpoints \
    "${COMMON[@]}" --progressive-proxy-mode static \
    --output-root "${RUN_ROOT}/runs/ewc_static48"

run_stage \
  plain_er_static48 \
  "${RUN_ROOT}/runs/plain_er_static48/metrics.json" \
  "${PYTHON}" experiments/replay_cl_eeg.py \
    --method plain_er --memory-capacity 1000 --replay-ratio 1.0 \
    "${COMMON[@]}" --progressive-proxy-mode static \
    --output-root "${RUN_ROOT}/runs/plain_er_static48"

run_stage \
  ewc_v2_s05_c20 \
  "${RUN_ROOT}/runs/ewc_v2_s05_c20/ewc/metrics.json" \
  "${PYTHON}" experiments/regularization_cl_eeg.py \
    --methods ewc --ewc-strength 5000 --no-save-checkpoints \
    "${COMMON[@]}" "${V2[@]}" \
    --output-root "${RUN_ROOT}/runs/ewc_v2_s05_c20"

run_stage \
  plain_er_v2_s05_c20 \
  "${RUN_ROOT}/runs/plain_er_v2_s05_c20/metrics.json" \
  "${PYTHON}" experiments/replay_cl_eeg.py \
    --method plain_er --memory-capacity 1000 --replay-ratio 1.0 \
    "${COMMON[@]}" "${V2[@]}" \
    --output-root "${RUN_ROOT}/runs/plain_er_v2_s05_c20"

run_stage \
  ewc_v2_s10_c40 \
  "${RUN_ROOT}/runs/ewc_v2_s10_c40/ewc/metrics.json" \
  "${PYTHON}" experiments/regularization_cl_eeg.py \
    --methods ewc --ewc-strength 5000 --no-save-checkpoints \
    "${COMMON[@]}" "${V2_STRONG[@]}" \
    --output-root "${RUN_ROOT}/runs/ewc_v2_s10_c40"

run_stage \
  plain_er_v2_s10_c40 \
  "${RUN_ROOT}/runs/plain_er_v2_s10_c40/metrics.json" \
  "${PYTHON}" experiments/replay_cl_eeg.py \
    --method plain_er --memory-capacity 1000 --replay-ratio 1.0 \
    "${COMMON[@]}" "${V2_STRONG[@]}" \
    --output-root "${RUN_ROOT}/runs/plain_er_v2_s10_c40"

run_stage \
  ewc_v2_s15_c60 \
  "${RUN_ROOT}/runs/ewc_v2_s15_c60/ewc/metrics.json" \
  "${PYTHON}" experiments/regularization_cl_eeg.py \
    --methods ewc --ewc-strength 5000 --no-save-checkpoints \
    "${COMMON[@]}" "${V2_STRONG_C60[@]}" \
    --output-root "${RUN_ROOT}/runs/ewc_v2_s15_c60"

run_stage \
  plain_er_v2_s15_c60 \
  "${RUN_ROOT}/runs/plain_er_v2_s15_c60/metrics.json" \
  "${PYTHON}" experiments/replay_cl_eeg.py \
    --method plain_er --memory-capacity 1000 --replay-ratio 1.0 \
    "${COMMON[@]}" "${V2_STRONG_C60[@]}" \
    --output-root "${RUN_ROOT}/runs/plain_er_v2_s15_c60"

"${PYTHON}" experiments/summarize_cross_cl_proxy_v2.py \
  --run-root "${RUN_ROOT}" \
  --clean-root "${CLEAN_ROOT}" \
  --bci "${BCI_PATH}" \
  | tee "${RUN_ROOT}/logs/summarize.log"

touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
