#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/population_persist_eeg_order_matrix_v1}"
CLEAN_ROOT="${CLEAN_ROOT:-${REPO_ROOT}/experiments/persist_eeg_order_matrix_v1/clean}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
FACED_ROOT="${FACED_ROOT:-/home/undefined/Disk/datasets/FACED_processed}"
SEEDS="${SEEDS:-4321,4322,4323}"
SCHEDULES="${SCHEDULES:-uniform_random,stratified_random,late_random}"
METHODS="${METHODS:-ewc,plain_er}"
GPUS="${GPUS:-0,1,2,3}"
DATASETS="${DATASETS:-ISRUC,FACED}"

PYTHON_SITE="$(${PYTHON} -c 'import site; print(site.getsitepackages()[0])')"
if [[ -d "${PYTHON_SITE}/nvidia" ]]; then
  NVIDIA_LIBRARY_PATH="$(find "${PYTHON_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)"
  export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runs"

PROXY_COMMON=(
  --progressive-proxy-mode population_feedback --progressive-persist
  --progressive-population-refresh-mix 0.20
  --progressive-population-candidates-per-class 4
  --progressive-direction-bank-capacity 4 --progressive-direction-bank-decay 0.8
  --progressive-proxy-lr 1e-6 --progressive-feedback-steps 4
  --progressive-feedback-batch 4 --progressive-guide-epochs 1
  --progressive-generation-steps 20 --progressive-generation-attempts 8
  --progressive-generation-batch 4 --progressive-reference-batch 16
  --progressive-step-relative-l2 0.05 --progressive-step-linf-std 0.25
  --progressive-cumulative-relative-l2 0.20
  --progressive-cumulative-linf-std 0.50 --progressive-history-weight 1
  --progressive-history-decay 0.8 --progressive-input-cone-residual 0.2
  --progressive-source-weight 0.5 --progressive-feedback-weight 1
  --progressive-feedback-decay 0.95 --progressive-feedback-capacity 2500
  --progressive-match-task-sequence-count --progressive-active-fraction 1
  --progressive-require-all-sequences-modified --progressive-fill-step-budget
  --progressive-history-refresh-count 16 --progressive-target-weight 200
  --progressive-conflict-weight 20 --progressive-gradient-norm-weight 1
  --progressive-virtual-old-weight 5 --progressive-virtual-new-weight 5
  --progressive-confidence-weight 0 --progressive-l2-weight 0
  --progressive-max-source-gradient-cosine 0.1
  --progressive-source-gate-samples 64 --progressive-survival-trajectories 2
  --progressive-survival-steps 2 --progressive-survival-batch 2
  --progressive-survival-weight 10 --progressive-survival-temperature 0.25
)

run_one() {
  local gpu="$1" dataset="$2" data_root="$3" total_tasks="$4"
  local method="$5" seed="$6" schedule="$7"
  local lower="${dataset,,}"
  local out="${RUN_ROOT}/runs/${lower}/${schedule}/${method}/seed${seed}"
  local evidence="${out}/summary.json"
  [[ -f "${evidence}" ]] && return
  local proxy_tasks clean_tasks log
  proxy_tasks="$(${PYTHON} "${REPO_ROOT}/experiments/dynamic_proxy_position_schedule.py" --dataset "${dataset}" --strength 5 --placement "${schedule}" --random-seed "${seed}" --field proxy)"
  clean_tasks="$(${PYTHON} "${REPO_ROOT}/experiments/dynamic_proxy_position_schedule.py" --dataset "${dataset}" --strength 5 --placement "${schedule}" --random-seed "${seed}" --field clean)"
  log="${RUN_ROOT}/logs/population_${lower}_${schedule}_${method}_seed${seed}.log"
  local common=(
    --dataset "${dataset}" --data-root "${data_root}"
    --input-checkpoint-root "${CHECKPOINT_ROOT}" --output-root "${out}"
    --seed "${seed}" --pretrain-seed 4321 --gpu 0 --batch 32 --num-worker 0
    --ssl-epoch 10 --incremental-epoch 10 --ssl-lr 1e-6 --cl-lr 1e-7
    --retention-milestones "5,${total_tasks}"
    --progressive-proxy-tasks "${proxy_tasks}"
    --progressive-clean-feedback-tasks "${clean_tasks}"
    "${PROXY_COMMON[@]}"
  )
  if [[ "${method}" == "ewc" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
      "${common[@]}" --methods ewc --ewc-strength 5000 --no-save-checkpoints \
      --delete-poisoned-inputs-after-task >"${log}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${REPO_ROOT}/experiments/replay_cl_eeg.py" \
      "${common[@]}" --method plain_er --memory-capacity 1000 --replay-ratio 1 \
      >"${log}" 2>&1
  fi
  test -f "${evidence}"
}

IFS=',' read -r -a seed_list <<< "${SEEDS}"
IFS=',' read -r -a schedule_list <<< "${SCHEDULES}"
IFS=',' read -r -a method_list <<< "${METHODS}"
IFS=',' read -r -a gpu_list <<< "${GPUS}"
IFS=',' read -r -a dataset_list <<< "${DATASETS}"
jobs=()
for dataset in "${dataset_list[@]}"; do
  if [[ "${dataset}" == "ISRUC" ]]; then
    data_root="${ISRUC_ROOT}"; total=49
  else
    data_root="${FACED_ROOT}"; total=61
  fi
  for seed in "${seed_list[@]}"; do
    for schedule in "${schedule_list[@]}"; do
      for method in "${method_list[@]}"; do
        jobs+=("${dataset}|${data_root}|${total}|${method}|${seed}|${schedule}")
      done
    done
  done
done

cd "${REPO_ROOT}"
pids=()
for lane in "${!gpu_list[@]}"; do
  (
    for ((index=lane; index<${#jobs[@]}; index+=${#gpu_list[@]})); do
      IFS='|' read -r dataset data_root total method seed schedule <<< "${jobs[index]}"
      run_one "${gpu_list[lane]}" "${dataset}" "${data_root}" "${total}" "${method}" "${seed}" "${schedule}"
    done
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  printf 'At least one population Proxy lane failed; see %s/logs.\n' "${RUN_ROOT}" >&2
  exit 1
fi
expected_runs=$((${#dataset_list[@]} * ${#seed_list[@]} * ${#schedule_list[@]} * ${#method_list[@]}))
completed_runs=$(find "${RUN_ROOT}/runs" -type f -name summary.json | wc -l)
if [[ "${completed_runs}" -lt "${expected_runs}" ]]; then
  printf 'Expected at least %d completed runs, found %d.\n' "${expected_runs}" "${completed_runs}" >&2
  exit 1
fi
touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
"${PYTHON}" "${REPO_ROOT}/experiments/summarize_population_persist_order_matrix.py" \
  --run-root "${RUN_ROOT}" --clean-root "${CLEAN_ROOT}"
