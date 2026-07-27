#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/progressive_feedback_proxy_runs/full_spr_puridiver_odd25_seed4321}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${REPO_ROOT}/experiments/progressive_feedback_proxy_runs/cross_cl_v2_full49_seed4321}"
BCI_PATH="${BCI_PATH:-/home/undefined/Desktop/IPhone/BCI.md}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runs"

PROXY_COMMON=(
  --progressive-proxy-tasks odd
  --progressive-clean-feedback-tasks even
  --progressive-base-subject 18
  --progressive-upload-full-pool
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
)

SPR_COMMON=(
  --gpu 0
  --max-subjects 0
  --ssl-epoch 10
  --expert-epochs 10
  --base-epochs 10
  --ft-epochs 10
  --batch 8
  --eval-batch 32
  --num-worker 0
  --delayed-capacity-sequences 32
  --memory-capacity-epochs 1000
  --freeze-spr-bn-stats
  --retention-milestones 10,25,49
  --no-save-state
)

PURIDIVER_COMMON=(
  --gpu 0
  --max-subjects 0
  --ssl-epoch 10
  --batch 8
  --online-batch-sequences 8
  --replay-batch-sequences 8
  --infer-batch 16
  --eval-batch 32
  --num-worker 0
  --memory-capacity-epochs 1000
  --replay-epochs 10
  --warmup-epochs 2
  --freeze-student-bn-stats
  --retention-milestones 10,25,49
  --no-save-state
)

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
    tail -100 "${log}" >&2
    return 1
  fi
}

cd "${REPO_ROOT}"

run_stage \
  full_spr_static48 \
  "${RUN_ROOT}/runs/full_spr_static48/metrics.json" \
  "${PYTHON}" experiments/full_spr_eeg_adapted.py \
    --output-root "${RUN_ROOT}/runs/full_spr_static48" \
    "${SPR_COMMON[@]}" "${PROXY_COMMON[@]}" \
    --progressive-proxy-mode static

run_stage \
  full_spr_v2 \
  "${RUN_ROOT}/runs/full_spr_v2/metrics.json" \
  "${PYTHON}" experiments/full_spr_eeg_adapted.py \
    --output-root "${RUN_ROOT}/runs/full_spr_v2" \
    "${SPR_COMMON[@]}" "${PROXY_COMMON[@]}" \
    --progressive-proxy-mode feedback

run_stage \
  full_puridiver_static48 \
  "${RUN_ROOT}/runs/full_puridiver_static48/metrics.json" \
  "${PYTHON}" experiments/full_puridiver_eeg_adapted.py \
    --output-root "${RUN_ROOT}/runs/full_puridiver_static48" \
    "${PURIDIVER_COMMON[@]}" "${PROXY_COMMON[@]}" \
    --progressive-proxy-mode static

run_stage \
  full_puridiver_v2 \
  "${RUN_ROOT}/runs/full_puridiver_v2/metrics.json" \
  "${PYTHON}" experiments/full_puridiver_eeg_adapted.py \
    --output-root "${RUN_ROOT}/runs/full_puridiver_v2" \
    "${PURIDIVER_COMMON[@]}" "${PROXY_COMMON[@]}" \
    --progressive-proxy-mode feedback

"${PYTHON}" experiments/summarize_full_spr_puridiver_proxy_v2.py \
  --run-root "${RUN_ROOT}" \
  --reference-root "${REFERENCE_ROOT}" \
  --bci "${BCI_PATH}" \
  >"${RUN_ROOT}/logs/summarize.log"

touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
printf 'full SPR/PuriDivER proxy v2 complete: %s\n' "${RUN_ROOT}"
