#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/cl_order_study_v1}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${RUN_ROOT}/manifests}"
CLEAN_ROOT="${CLEAN_ROOT:-${REPO_ROOT}/experiments/persist_eeg_order_matrix_v1/clean}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
FACED_ROOT="${FACED_ROOT:-/home/undefined/Disk/datasets/FACED_processed}"
DATASETS="${DATASETS:-ISRUC,FACED}"
ORDERS="${ORDERS:-seed_random,easy_to_hard,hard_to_easy,source_near_to_far,smooth_nearest,diversity_greedy}"
METHODS="${METHODS:-ewc,plain_er}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-4321}"

PYTHON_SITE="$(${PYTHON} -c 'import site; print(site.getsitepackages()[0])')"
if [[ -d "${PYTHON_SITE}/nvidia" ]]; then
  NVIDIA_LIBRARY_PATH="$(find "${PYTHON_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)"
  export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runs"

run_one() {
  local gpu="$1" dataset="$2" data_root="$3" total="$4" order="$5" method="$6"
  local lower="${dataset,,}"
  local out="${RUN_ROOT}/runs/${lower}/${order}/${method}/seed${SEED}"
  local evidence="${out}/summary.json"
  [[ -f "${evidence}" ]] && return
  mkdir -p "$(dirname "${out}")"
  if [[ "${order}" == "seed_random" ]]; then
    cp -a "${CLEAN_ROOT}/${lower}/${method}/seed${SEED}" "${out}"
    test -f "${evidence}"
    return
  fi
  local manifest="${MANIFEST_ROOT}/${lower}/seed${SEED}/${order}.json"
  local log="${RUN_ROOT}/logs/${lower}_${order}_${method}_seed${SEED}.log"
  local common=(
    --dataset "${dataset}" --data-root "${data_root}"
    --input-checkpoint-root "${CHECKPOINT_ROOT}" --output-root "${out}"
    --seed "${SEED}" --pretrain-seed 4321 --subject-order-manifest "${manifest}"
    --gpu 0 --batch 32 --num-worker 0 --ssl-epoch 10 --incremental-epoch 10
    --ssl-lr 1e-6 --cl-lr 1e-7 --retention-milestones "5,${total}"
  )
  if [[ "${method}" == "ewc" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
      "${common[@]}" --methods ewc --ewc-strength 5000 --no-save-checkpoints \
      >"${log}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${REPO_ROOT}/experiments/replay_cl_eeg.py" \
      "${common[@]}" --method plain_er --memory-capacity 1000 --replay-ratio 1 \
      >"${log}" 2>&1
  fi
  test -f "${evidence}"
}

IFS=',' read -r -a dataset_list <<< "${DATASETS}"
IFS=',' read -r -a order_list <<< "${ORDERS}"
IFS=',' read -r -a method_list <<< "${METHODS}"
IFS=',' read -r -a gpu_list <<< "${GPUS}"
jobs=()
for dataset in "${dataset_list[@]}"; do
  if [[ "${dataset}" == "ISRUC" ]]; then
    data_root="${ISRUC_ROOT}"; total=49
  else
    data_root="${FACED_ROOT}"; total=61
  fi
  for order in "${order_list[@]}"; do
    for method in "${method_list[@]}"; do
      jobs+=("${dataset}|${data_root}|${total}|${order}|${method}")
    done
  done
done

cd "${REPO_ROOT}"
pids=()
for lane in "${!gpu_list[@]}"; do
  (
    for ((index=lane; index<${#jobs[@]}; index+=${#gpu_list[@]})); do
      IFS='|' read -r dataset data_root total order method <<< "${jobs[index]}"
      run_one "${gpu_list[lane]}" "${dataset}" "${data_root}" "${total}" "${order}" "${method}"
    done
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || exit 1
expected=$((${#dataset_list[@]} * ${#order_list[@]} * ${#method_list[@]}))
completed=$(find "${RUN_ROOT}/runs" -name summary.json | wc -l)
[[ "${completed}" -ge "${expected}" ]]
"${PYTHON}" "${REPO_ROOT}/experiments/summarize_cl_order_study.py" \
  --run-root "${RUN_ROOT}" --manifest-root "${MANIFEST_ROOT}"
touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
