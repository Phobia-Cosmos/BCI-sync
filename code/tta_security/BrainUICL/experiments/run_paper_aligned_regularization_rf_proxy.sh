#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments/paper_aligned_regularization_rf_proxy}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
FACED_ROOT="${FACED_ROOT:-/home/undefined/Disk/datasets/FACED_processed}"
PROXY_ROOT="${PROXY_ROOT:-${REPO_ROOT}/experiments/frozen_proxy_frequency_shift/full49_seed4321/rel_l2_0500/F-S}"
PHASE="${PHASE:-all}"

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
  local condition="$3"
  local defense_mode="$4"
  shift 4
  local output_dir="${OUTPUT_ROOT}/${dataset,,}/${condition}"
  local log_path="${OUTPUT_ROOT}/${dataset,,}/${condition}.log"

  mkdir -p "$(dirname "${output_dir}")"
  if summary_complete "${output_dir}/summary.json"; then
    printf '[skip] %s/%s is complete\n' "${dataset}" "${condition}"
    return
  fi

  printf '[run] %s/%s\n' "${dataset}" "${condition}"
  "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
    --dataset "${dataset}" \
    --data-root "${data_root}" \
    --output-root "${output_dir}" \
    --methods "${METHODS}" \
    --defense-mode "${defense_mode}" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    > "${log_path}" 2>&1
}

run_clean() {
  run_one ISRUC "${ISRUC_ROOT}" clean_none none
  run_one ISRUC "${ISRUC_ROOT}" clean_robust_feature robust_feature
  run_one FACED "${FACED_ROOT}" clean_none none
  run_one FACED "${FACED_ROOT}" clean_robust_feature robust_feature
}

run_proxy() {
  if [[ ! -d "${PROXY_ROOT}" ]]; then
    printf 'Missing fixed Proxy stream: %s\n' "${PROXY_ROOT}" >&2
    exit 1
  fi
  run_one ISRUC "${ISRUC_ROOT}" proxy_none none \
    --noise-upload-root "${PROXY_ROOT}"
  run_one ISRUC "${ISRUC_ROOT}" proxy_robust_feature robust_feature \
    --noise-upload-root "${PROXY_ROOT}"
}

case "${PHASE}" in
  clean)
    run_clean
    ;;
  proxy)
    run_proxy
    ;;
  all)
    run_clean
    run_proxy
    ;;
  *)
    printf 'PHASE must be clean, proxy, or all\n' >&2
    exit 2
    ;;
esac

printf '[complete] phase=%s\n' "${PHASE}"
