#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/persist_eeg_probe_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
FACED_ROOT="${FACED_ROOT:-/home/undefined/Disk/datasets/FACED_processed}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runs"

PROXY_COMMON=(
  --progressive-proxy-mode feedback
  --progressive-persist
  --progressive-direction-bank-capacity 4
  --progressive-direction-bank-decay 0.8
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
  --progressive-history-weight 1
  --progressive-history-decay 0.8
  --progressive-input-cone-residual 0.2
  --progressive-source-weight 0.5
  --progressive-feedback-weight 1
  --progressive-feedback-decay 0.95
  --progressive-feedback-capacity 2500
  --progressive-match-task-sequence-count
  --progressive-active-fraction 1
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
  --progressive-survival-trajectories 2
  --progressive-survival-steps 2
  --progressive-survival-batch 2
  --progressive-survival-weight 10
  --progressive-survival-temperature 0.25
)

run_one() {
  local dataset="$1" data_root="$2" total_tasks="$3" method="$4"
  local lower="${dataset,,}"
  local out="${RUN_ROOT}/runs/${lower}/${method}"
  local proxy_tasks clean_tasks
  proxy_tasks="$(${PYTHON} "${REPO_ROOT}/experiments/dynamic_proxy_position_schedule.py" --dataset "${dataset}" --strength 5 --placement random --random-seed 4321 --field proxy)"
  clean_tasks="$(${PYTHON} "${REPO_ROOT}/experiments/dynamic_proxy_position_schedule.py" --dataset "${dataset}" --strength 5 --placement random --random-seed 4321 --field clean)"
  local common=(
    --dataset "${dataset}" --data-root "${data_root}" --input-checkpoint-root "${CHECKPOINT_ROOT}"
    --output-root "${out}" --seed 4321 --gpu 0 --batch 32 --num-worker 0
    --ssl-epoch 10 --incremental-epoch 10 --ssl-lr 1e-6 --cl-lr 1e-7
    --retention-milestones "5,${total_tasks}"
    --progressive-proxy-tasks "${proxy_tasks}" --progressive-clean-feedback-tasks "${clean_tasks}"
    "${PROXY_COMMON[@]}"
  )
  if [[ "${method}" == "ewc" ]]; then
    "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" "${common[@]}" --methods ewc \
      --ewc-strength 5000 --no-save-checkpoints --delete-poisoned-inputs-after-task \
      >"${RUN_ROOT}/logs/${lower}_${method}.log" 2>&1
  else
    "${PYTHON}" "${REPO_ROOT}/experiments/replay_cl_eeg.py" "${common[@]}" \
      --method plain_er --memory-capacity 1000 --replay-ratio 1 \
      >"${RUN_ROOT}/logs/${lower}_${method}.log" 2>&1
  fi
}

cd "${REPO_ROOT}"
run_one ISRUC "${ISRUC_ROOT}" 49 ewc
run_one ISRUC "${ISRUC_ROOT}" 49 plain_er
run_one FACED "${FACED_ROOT}" 61 ewc
run_one FACED "${FACED_ROOT}" 61 plain_er
touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
printf '[complete] PERSIST-EEG probe: %s\n' "${RUN_ROOT}"
