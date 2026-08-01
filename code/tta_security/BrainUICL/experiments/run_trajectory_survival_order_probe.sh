#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/trajectory_survival_order_probe_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
FACED_ROOT="${FACED_ROOT:-/home/undefined/Disk/datasets/FACED_processed}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runs"

REGULARIZATION_COMMON=(
  --seed 4321
  --gpu 0
  --batch 32
  --num-worker 0
  --ssl-epoch 10
  --incremental-epoch 10
  --ssl-lr 1e-6
  --cl-lr 1e-7
  --ewc-strength 5000
  --input-checkpoint-root "${CHECKPOINT_ROOT}"
  --no-save-checkpoints
  --delete-poisoned-inputs-after-task
)

FULL_COMMON=(
  --seed 4321
  --gpu 0
  --batch 32
  --eval-batch 32
  --num-worker 0
  --max-subjects 0
  --ssl-epoch 10
  --lr 1e-4
  --ssl-lr 1e-6
  --beta1 0.5
  --beta2 0.99
  --weight-decay 3e-4
  --grad-clip 5
  --input-checkpoint-root "${CHECKPOINT_ROOT}"
  --no-save-state
)

SPR_COMMON=(
  --expert-epochs 10
  --base-epochs 10
  --ft-epochs 10
  --spr-ssl-lr 1e-6
  --ft-lr 1e-7
  --delayed-capacity-sequences 32
  --memory-capacity-epochs 1000
)

# This is the only changed Proxy objective relative to the completed matrix.
# The repair data are source-pretrain data and never future incremental users.
PROXY_COMMON=(
  --progressive-proxy-mode feedback
  --progressive-base-subject 18
  --progressive-proxy-lr 1e-6
  --progressive-feedback-steps 4
  --progressive-feedback-batch 4
  --progressive-guide-epochs 1
  --progressive-generation-steps 20
  --progressive-generation-attempts 1
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
  --progressive-survival-trajectories 3
  --progressive-survival-steps 3
  --progressive-survival-batch 2
  --progressive-survival-weight 20
  --progressive-survival-temperature 0.25
)

run_stage() {
  local name="$1"
  local evidence="$2"
  shift 2
  local marker="${RUN_ROOT}/.${name}.complete"
  local log="${RUN_ROOT}/logs/${name}.log"
  if [[ -f "${marker}" && -f "${evidence}" ]]; then
    printf '[skip] %s\n' "${name}"
    return
  fi
  local started="${SECONDS}"
  printf '[run] %s\n' "${name}"
  if "$@" >"${log}" 2>&1; then
    test -f "${evidence}"
    touch "${marker}"
    printf '[done] %s (%ss)\n' "${name}" "$((SECONDS - started))"
  else
    tail -100 "${log}" >&2
    return 1
  fi
}

run_condition() {
  local dataset="$1"
  local data_root="$2"
  local total_tasks="$3"
  local placement="$4"
  local lower="${dataset,,}"
  local condition="k10_${placement}"
  local root="${RUN_ROOT}/runs/${lower}/${condition}"
  local proxy_tasks
  local clean_tasks
  proxy_tasks="$("${PYTHON}" "${REPO_ROOT}/experiments/dynamic_proxy_position_schedule.py" \
    --dataset "${dataset}" --strength 10 --placement "${placement}" --field proxy)"
  clean_tasks="$("${PYTHON}" "${REPO_ROOT}/experiments/dynamic_proxy_position_schedule.py" \
    --dataset "${dataset}" --strength 10 --placement "${placement}" --field clean)"
  local schedule_args=(
    --progressive-proxy-tasks "${proxy_tasks}"
    --progressive-clean-feedback-tasks "${clean_tasks}"
  )
  local retention="10,20,25,${total_tasks}"

  run_stage \
    "${lower}_${condition}_ewc" \
    "${root}/regularization/ewc/metrics.json" \
    "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
      --dataset "${dataset}" --data-root "${data_root}" \
      --output-root "${root}/regularization" \
      --methods ewc --defense-mode none \
      --retention-milestones "${retention}" \
      "${REGULARIZATION_COMMON[@]}" "${PROXY_COMMON[@]}" "${schedule_args[@]}"

  run_stage \
    "${lower}_${condition}_full_spr" \
    "${root}/full_spr/metrics.json" \
    "${PYTHON}" "${REPO_ROOT}/experiments/full_spr_eeg_adapted.py" \
      --dataset "${dataset}" --data-root "${data_root}" \
      --output-root "${root}/full_spr" \
      --retention-milestones "${retention}" \
      "${FULL_COMMON[@]}" "${SPR_COMMON[@]}" \
      "${PROXY_COMMON[@]}" "${schedule_args[@]}"
}

cd "${REPO_ROOT}"
run_condition ISRUC "${ISRUC_ROOT}" 49 front
run_condition ISRUC "${ISRUC_ROOT}" 49 middle
run_condition ISRUC "${ISRUC_ROOT}" 49 tail
run_condition FACED "${FACED_ROOT}" 61 front
run_condition FACED "${FACED_ROOT}" 61 middle
run_condition FACED "${FACED_ROOT}" 61 tail

"${PYTHON}" "${REPO_ROOT}/experiments/summarize_trajectory_survival_order_probe.py" \
  --run-root "${RUN_ROOT}" \
  --completed-matrix experiments/dynamic_proxy_position_matrix_v1/RESULTS.json \
  --regularization-clean-isruc \
    experiments/paper_aligned_regularization_clean/isruc_b32_bn_updated_lr1e7_clean_full49_seed4321/ewc/metrics.json \
  --regularization-clean-faced \
    experiments/paper_aligned_regularization_clean/faced_b32_bn_updated_lr1e7_clean_full61_seed4321/ewc/metrics.json \
  --replay-clean-isruc experiments/dynamic_proxy_position_matrix_v1/clean/isruc/full_spr/metrics.json \
  --replay-clean-faced experiments/dynamic_proxy_position_matrix_v1/clean/faced/full_spr/metrics.json
touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
printf '[complete] trajectory survival order probe: %s\n' "${RUN_ROOT}"
