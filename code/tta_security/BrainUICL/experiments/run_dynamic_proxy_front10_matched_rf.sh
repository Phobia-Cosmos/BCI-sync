#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments/dynamic_proxy_front10_matched_rf}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
FACED_ROOT="${FACED_ROOT:-/home/undefined/Disk/datasets/FACED_processed}"

METHODS="finetune,ewc,online_ewc,si,mas"
PROXY_TASKS="1,3,5,7,9,11,13,15,17,19"
ISRUC_CLEAN_TASKS="2,4,6,8,10,12,14,16,18,$(seq -s, 20 49)"
FACED_CLEAN_TASKS="2,4,6,8,10,12,14,16,18,$(seq -s, 20 61)"

COMMON_ARGS=(
  --seed 4321
  --gpu 0
  --batch 32
  --num-worker 0
  --ssl-epoch 10
  --incremental-epoch 10
  --ssl-lr 1e-6
  --cl-lr 1e-7
  --ewc-strength 5000
  --online-ewc-strength 6500
  --online-ewc-decay 1.0
  --si-strength 1500000
  --si-xi 1e-6
  --mas-strength 3000
  --mas-decay 1.0
  --input-checkpoint-root "${CHECKPOINT_ROOT}"
  --no-save-checkpoints
  --progressive-proxy-mode feedback
  --progressive-proxy-tasks "${PROXY_TASKS}"
  --progressive-proxy-lr 1e-6
  --progressive-feedback-steps 4
  --progressive-feedback-batch 4
  --progressive-guide-epochs 1
  --progressive-generation-steps 20
  --progressive-generation-attempts 32
  --progressive-generation-batch 4
  --progressive-reference-batch 16
  --progressive-step-relative-l2 0.05
  --progressive-step-linf-std 0.25
  --progressive-cumulative-relative-l2 0.20
  --progressive-cumulative-linf-std 0.50
  --progressive-history-weight 1.0
  --progressive-history-decay 0.8
  --progressive-input-cone-residual 0.2
  --progressive-source-weight 0.5
  --progressive-feedback-weight 1.0
  --progressive-feedback-decay 0.95
  --progressive-feedback-capacity 2500
  --progressive-match-task-sequence-count
  --progressive-active-fraction 1.0
  --progressive-require-all-sequences-modified
  --progressive-fill-step-budget
  --progressive-history-refresh-count 16
  --progressive-target-weight 200
  --progressive-conflict-weight 20
  --progressive-gradient-norm-weight 1
  --progressive-virtual-old-weight 5
  --progressive-virtual-new-weight 5
  --progressive-confidence-weight 0
  --progressive-l2-weight 0
  --progressive-max-source-gradient-cosine 0.1
  --progressive-source-gate-samples 64
  --progressive-require-source-conflict
)

summary_complete() {
  local summary_path="$1"
  [[ -f "${summary_path}" ]] && \
    jq -e '(["finetune", "ewc", "online_ewc", "si", "mas"] - keys) | length == 0' \
      "${summary_path}" >/dev/null
}

run_one() {
  local dataset="$1"
  local data_root="$2"
  local base_subject="$3"
  local clean_tasks="$4"
  local defense="$5"
  local output_dir="${OUTPUT_ROOT}/${dataset,,}/${defense}"
  local log_path="${OUTPUT_ROOT}/${dataset,,}/${defense}.log"

  mkdir -p "$(dirname "${output_dir}")"
  if summary_complete "${output_dir}/summary.json"; then
    printf '[skip] %s/%s is complete\n' "${dataset}" "${defense}"
    return
  fi
  printf '[run] %s/%s\n' "${dataset}" "${defense}"
  "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
    --dataset "${dataset}" \
    --data-root "${data_root}" \
    --output-root "${output_dir}" \
    --methods "${METHODS}" \
    --defense-mode "${defense}" \
    --progressive-base-subject "${base_subject}" \
    --progressive-clean-feedback-tasks "${clean_tasks}" \
    "${COMMON_ARGS[@]}" \
    > "${log_path}" 2>&1
}

run_one ISRUC "${ISRUC_ROOT}" 18 "${ISRUC_CLEAN_TASKS}" none
run_one ISRUC "${ISRUC_ROOT}" 18 "${ISRUC_CLEAN_TASKS}" robust_feature
run_one FACED "${FACED_ROOT}" 18 "${FACED_CLEAN_TASKS}" none
run_one FACED "${FACED_ROOT}" 18 "${FACED_CLEAN_TASKS}" robust_feature

printf '[complete] front10 task-matched dynamic Proxy matrix\n'
