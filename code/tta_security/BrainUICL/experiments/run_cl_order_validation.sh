#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/cl_order_validation_v1}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${REPO_ROOT}/experiments/cl_order_study_v1/manifests}"
RANDOM_CLEAN_ROOT="${RANDOM_CLEAN_ROOT:-${REPO_ROOT}/experiments/persist_eeg_order_matrix_v1/clean}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
GPUS="${GPUS:-0,3,4}"

PYTHON_SITE="$(${PYTHON} -c 'import site; print(site.getsitepackages()[0])')"
if [[ -d "${PYTHON_SITE}/nvidia" ]]; then
  NVIDIA_LIBRARY_PATH="$(find "${PYTHON_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)"
  export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runs"

jobs=(
  "4322|ewc|hard_to_easy" "4322|plain_er|source_near_to_far"
  "4323|ewc|hard_to_easy" "4323|plain_er|source_near_to_far"
)
run_one() {
  local gpu="$1" seed="$2" method="$3" order="$4"
  local out="${RUN_ROOT}/runs/isruc/${order}/${method}/seed${seed}"
  [[ -f "${out}/summary.json" ]] && return
  local manifest="${MANIFEST_ROOT}/isruc/seed${seed}/${order}.json"
  local common=(
    --dataset ISRUC --data-root "${ISRUC_ROOT}" --input-checkpoint-root "${CHECKPOINT_ROOT}"
    --output-root "${out}" --seed "${seed}" --pretrain-seed 4321
    --subject-order-manifest "${manifest}" --gpu 0 --batch 32 --num-worker 0
    --ssl-epoch 10 --incremental-epoch 10 --ssl-lr 1e-6 --cl-lr 1e-7
    --retention-milestones 5,49
  )
  if [[ "${method}" == "ewc" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
      "${common[@]}" --methods ewc --ewc-strength 5000 --no-save-checkpoints \
      >"${RUN_ROOT}/logs/seed${seed}_${method}_${order}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${REPO_ROOT}/experiments/replay_cl_eeg.py" \
      "${common[@]}" --method plain_er --memory-capacity 1000 --replay-ratio 1 \
      >"${RUN_ROOT}/logs/seed${seed}_${method}_${order}.log" 2>&1
  fi
}

IFS=',' read -r -a gpu_list <<< "${GPUS}"
pids=()
for lane in "${!gpu_list[@]}"; do
  (
    for ((index=lane; index<${#jobs[@]}; index+=${#gpu_list[@]})); do
      IFS='|' read -r seed method order <<< "${jobs[index]}"
      run_one "${gpu_list[lane]}" "${seed}" "${method}" "${order}"
    done
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]]
"${PYTHON}" "${REPO_ROOT}/experiments/summarize_cl_order_validation.py" \
  --run-root "${RUN_ROOT}" --random-clean-root "${RANDOM_CLEAN_ROOT}"
touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
