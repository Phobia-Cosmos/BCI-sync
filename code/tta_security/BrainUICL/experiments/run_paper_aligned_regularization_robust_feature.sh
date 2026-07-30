#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments/paper_aligned_regularization_clean}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
FACED_ROOT="${FACED_ROOT:-/home/undefined/Disk/datasets/FACED_processed}"

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

run_one() {
  local dataset="$1"
  local data_root="$2"
  local condition="$3"
  local methods="$4"
  local defense_mode="$5"
  local task_count="$6"
  local output_dir="${OUTPUT_ROOT}/${dataset,,}_b32_bn_updated_lr1e7_${condition}_full${task_count}_seed4321"

  if [[ -f "${output_dir}/summary.json" ]]; then
    echo "Skipping completed run: ${output_dir}"
    return
  fi

  mkdir -p "${output_dir}"
  "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
    --dataset "${dataset}" \
    --data-root "${data_root}" \
    --output-root "${output_dir}" \
    --methods "${methods}" \
    --defense-mode "${defense_mode}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${output_dir}.log"
}

run_one ISRUC "${ISRUC_ROOT}" clean "finetune,ewc,online_ewc,si,mas" none 49
run_one ISRUC "${ISRUC_ROOT}" ewc_robust_feature "ewc" robust_feature 49
run_one FACED "${FACED_ROOT}" clean "finetune,ewc,online_ewc,si,mas" none 61
run_one FACED "${FACED_ROOT}" ewc_robust_feature "ewc" robust_feature 61
