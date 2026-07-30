#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments/dynamic_proxy_all_sequences_rf}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
FACED_ROOT="${FACED_ROOT:-/home/undefined/Disk/datasets/FACED_processed}"

METHODS="finetune,ewc,online_ewc,si,mas"
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
  --progressive-proxy-tasks odd
  --progressive-clean-feedback-tasks even
  --progressive-base-subject 18
  --progressive-proxy-lr 1e-6
  --progressive-feedback-steps 4
  --progressive-feedback-batch 4
  --progressive-guide-epochs 1
  --progressive-generation-steps 20
  --progressive-generation-attempts 8
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
  --progressive-upload-full-pool
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
  local defense="$3"
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
    "${COMMON_ARGS[@]}" \
    > "${log_path}" 2>&1
}

run_one ISRUC "${ISRUC_ROOT}" none
run_one ISRUC "${ISRUC_ROOT}" robust_feature
run_one FACED "${FACED_ROOT}" none
run_one FACED "${FACED_ROOT}" robust_feature

printf '[complete] dynamic Proxy full matrix\n'
