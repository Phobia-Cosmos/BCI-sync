#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/dynamic_proxy_position_matrix_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
FACED_ROOT="${FACED_ROOT:-/home/undefined/Disk/datasets/FACED_processed}"
REG_CLEAN_ISRUC="${REG_CLEAN_ISRUC:-${REPO_ROOT}/experiments/paper_aligned_regularization_clean/isruc_b32_bn_updated_lr1e7_clean_full49_seed4321}"
REG_CLEAN_FACED="${REG_CLEAN_FACED:-${REPO_ROOT}/experiments/paper_aligned_regularization_clean/faced_b32_bn_updated_lr1e7_clean_full61_seed4321}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/clean" "${RUN_ROOT}/runs"
"${PYTHON}" "${REPO_ROOT}/experiments/dynamic_proxy_position_schedule.py" \
  --field manifest --output "${RUN_ROOT}/SCHEDULES.json"

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
  --online-ewc-strength 6500
  --online-ewc-decay 1.0
  --si-strength 1500000
  --si-xi 1e-6
  --mas-strength 3000
  --mas-decay 1.0
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

PURIDIVER_COMMON=(
  --online-batch-sequences 8
  --replay-batch-sequences 8
  --infer-batch 16
  --cl-lr 1e-7
  --memory-capacity-epochs 1000
  --replay-epochs 10
  --warmup-epochs 2
)

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

dataset_values() {
  local dataset="$1"
  if [[ "${dataset}" == "ISRUC" ]]; then
    printf '%s\n' "${ISRUC_ROOT}" 49
  else
    printf '%s\n' "${FACED_ROOT}" 61
  fi
}

run_clean_full_methods() {
  local dataset="$1"
  local data_root="$2"
  local total_tasks="$3"
  local lower="${dataset,,}"
  local retention="10,20,25,${total_tasks}"

  run_stage \
    "${lower}_clean_full_spr" \
    "${RUN_ROOT}/clean/${lower}/full_spr/metrics.json" \
    "${PYTHON}" "${REPO_ROOT}/experiments/full_spr_eeg_adapted.py" \
      --dataset "${dataset}" --data-root "${data_root}" \
      --output-root "${RUN_ROOT}/clean/${lower}/full_spr" \
      --retention-milestones "${retention}" \
      --progressive-proxy-mode none \
      "${FULL_COMMON[@]}" "${SPR_COMMON[@]}"

  run_stage \
    "${lower}_clean_full_puridiver" \
    "${RUN_ROOT}/clean/${lower}/full_puridiver/metrics.json" \
    "${PYTHON}" "${REPO_ROOT}/experiments/full_puridiver_eeg_adapted.py" \
      --dataset "${dataset}" --data-root "${data_root}" \
      --output-root "${RUN_ROOT}/clean/${lower}/full_puridiver" \
      --retention-milestones "${retention}" \
      --progressive-proxy-mode none \
      "${FULL_COMMON[@]}" "${PURIDIVER_COMMON[@]}"
}

run_proxy_condition() {
  local dataset="$1"
  local data_root="$2"
  local total_tasks="$3"
  local strength="$4"
  local placement="$5"
  local lower="${dataset,,}"
  local condition="k${strength}_${placement}"
  local condition_root="${RUN_ROOT}/runs/${lower}/${condition}"
  local proxy_tasks
  local clean_tasks
  proxy_tasks="$("${PYTHON}" "${REPO_ROOT}/experiments/dynamic_proxy_position_schedule.py" \
    --dataset "${dataset}" --strength "${strength}" \
    --placement "${placement}" --field proxy)"
  clean_tasks="$("${PYTHON}" "${REPO_ROOT}/experiments/dynamic_proxy_position_schedule.py" \
    --dataset "${dataset}" --strength "${strength}" \
    --placement "${placement}" --field clean)"
  local schedule_args=(
    --progressive-proxy-tasks "${proxy_tasks}"
    --progressive-clean-feedback-tasks "${clean_tasks}"
  )
  local retention="10,20,25,${total_tasks}"

  run_stage \
    "${lower}_${condition}_regularization" \
    "${condition_root}/regularization/summary.json" \
    "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
      --dataset "${dataset}" --data-root "${data_root}" \
      --output-root "${condition_root}/regularization" \
      --methods finetune,ewc,online_ewc,si,mas \
      --defense-mode none --retention-milestones "${retention}" \
      "${REGULARIZATION_COMMON[@]}" "${PROXY_COMMON[@]}" "${schedule_args[@]}"

  run_stage \
    "${lower}_${condition}_full_spr" \
    "${condition_root}/full_spr/metrics.json" \
    "${PYTHON}" "${REPO_ROOT}/experiments/full_spr_eeg_adapted.py" \
      --dataset "${dataset}" --data-root "${data_root}" \
      --output-root "${condition_root}/full_spr" \
      --retention-milestones "${retention}" \
      "${FULL_COMMON[@]}" "${SPR_COMMON[@]}" \
      "${PROXY_COMMON[@]}" "${schedule_args[@]}"

  run_stage \
    "${lower}_${condition}_full_puridiver" \
    "${condition_root}/full_puridiver/metrics.json" \
    "${PYTHON}" "${REPO_ROOT}/experiments/full_puridiver_eeg_adapted.py" \
      --dataset "${dataset}" --data-root "${data_root}" \
      --output-root "${condition_root}/full_puridiver" \
      --retention-milestones "${retention}" \
      "${FULL_COMMON[@]}" "${PURIDIVER_COMMON[@]}" \
      "${PROXY_COMMON[@]}" "${schedule_args[@]}"
}

cd "${REPO_ROOT}"
for dataset in ISRUC FACED; do
  mapfile -t values < <(dataset_values "${dataset}")
  data_root="${values[0]}"
  total_tasks="${values[1]}"
  run_clean_full_methods "${dataset}" "${data_root}" "${total_tasks}"
  for strength in 10 20; do
    for placement in front middle tail; do
      run_proxy_condition \
        "${dataset}" "${data_root}" "${total_tasks}" "${strength}" "${placement}"
    done
  done
done

"${PYTHON}" "${REPO_ROOT}/experiments/summarize_dynamic_proxy_position_matrix.py" \
  --run-root "${RUN_ROOT}" \
  --regularization-clean-isruc "${REG_CLEAN_ISRUC}" \
  --regularization-clean-faced "${REG_CLEAN_FACED}" \
  >"${RUN_ROOT}/logs/summarize.log"
touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
printf '[complete] dynamic Proxy position matrix: %s\n' "${RUN_ROOT}"
