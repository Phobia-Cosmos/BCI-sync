#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/invariant_proxy_order_study_v1}"
ORDER_CLEAN_ROOT="${ORDER_CLEAN_ROOT:-${REPO_ROOT}/experiments/cl_order_study_v1}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ORDER_CLEAN_ROOT}/manifests}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/undefined/Disk/ai-storage/BrainUICL/model_parameter}"
ISRUC_ROOT="${ISRUC_ROOT:-/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32}"
ORDERS="${ORDERS:-seed_random,easy_to_hard,hard_to_easy,source_near_to_far,smooth_nearest,diversity_greedy}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-4321}"
PROXY_TASKS="${PROXY_TASKS:-9,19,26,34,40}"

PYTHON_SITE="$(${PYTHON} -c 'import site; print(site.getsitepackages()[0])')"
if [[ -d "${PYTHON_SITE}/nvidia" ]]; then
  NVIDIA_LIBRARY_PATH="$(find "${PYTHON_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)"
  export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runs"

clean_tasks="$(${PYTHON} - "${PROXY_TASKS}" <<'PY'
import sys
proxy={int(value) for value in sys.argv[1].split(',')}
print(','.join(str(task) for task in range(1,50) if task not in proxy))
PY
)"

run_one() {
  local gpu="$1" order="$2"
  local out="${RUN_ROOT}/runs/isruc/${order}/ewc/seed${SEED}"
  [[ -f "${out}/summary.json" ]] && return
  local manifest="${MANIFEST_ROOT}/isruc/seed${SEED}/${order}.json"
  local log="${RUN_ROOT}/logs/isruc_${order}_ewc_seed${SEED}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
    --dataset ISRUC --data-root "${ISRUC_ROOT}" \
    --input-checkpoint-root "${CHECKPOINT_ROOT}" --output-root "${out}" \
    --seed "${SEED}" --pretrain-seed 4321 --subject-order-manifest "${manifest}" \
    --gpu 0 --batch 32 --num-worker 0 --ssl-epoch 10 --incremental-epoch 10 \
    --ssl-lr 1e-6 --cl-lr 1e-7 --retention-milestones 5,49 \
    --methods ewc --ewc-strength 5000 --no-save-checkpoints \
    --delete-poisoned-inputs-after-task \
    --progressive-proxy-mode population_feedback --progressive-persist \
    --progressive-proxy-tasks "${PROXY_TASKS}" \
    --progressive-clean-feedback-tasks "${clean_tasks}" \
    --progressive-population-refresh-mix 0.20 \
    --progressive-population-candidates-per-class 4 \
    --progressive-population-cross-class-mix 0.35 \
    --progressive-preserve-eeg-invariants \
    --progressive-invariant-drift-tolerance 0.02 \
    --progressive-direction-bank-capacity 4 --progressive-direction-bank-decay 0.8 \
    --progressive-proxy-lr 1e-6 --progressive-feedback-steps 4 \
    --progressive-feedback-batch 4 --progressive-guide-epochs 1 \
    --progressive-generation-steps 20 --progressive-generation-attempts 8 \
    --progressive-generation-batch 4 --progressive-reference-batch 16 \
    --progressive-step-relative-l2 0.05 --progressive-step-linf-std 0.25 \
    --progressive-cumulative-relative-l2 0.20 --progressive-cumulative-linf-std 0.50 \
    --progressive-history-weight 1 --progressive-history-decay 0.8 \
    --progressive-input-cone-residual 0.2 --progressive-source-weight 0.5 \
    --progressive-feedback-weight 1 --progressive-feedback-decay 0.95 \
    --progressive-feedback-capacity 2500 --progressive-match-task-sequence-count \
    --progressive-active-fraction 1 --progressive-require-all-sequences-modified \
    --progressive-fill-step-budget --progressive-history-refresh-count 16 \
    --progressive-target-weight 200 --progressive-conflict-weight 20 \
    --progressive-gradient-norm-weight 1 --progressive-virtual-old-weight 5 \
    --progressive-virtual-new-weight 5 --progressive-confidence-weight 0 \
    --progressive-l2-weight 0 --progressive-max-source-gradient-cosine 0.1 \
    --progressive-source-gate-samples 64 --progressive-survival-trajectories 2 \
    --progressive-survival-steps 2 --progressive-survival-batch 2 \
    --progressive-survival-weight 10 --progressive-survival-temperature 0.25 \
    >"${log}" 2>&1
  test -f "${out}/summary.json"
}

IFS=',' read -r -a order_list <<< "${ORDERS}"
IFS=',' read -r -a gpu_list <<< "${GPUS}"
pids=()
for lane in "${!gpu_list[@]}"; do
  (
    for ((index=lane; index<${#order_list[@]}; index+=${#gpu_list[@]})); do
      run_one "${gpu_list[lane]}" "${order_list[index]}"
    done
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || exit 1
[[ "$(find "${RUN_ROOT}/runs" -name summary.json | wc -l)" -ge "${#order_list[@]}" ]]
"${PYTHON}" "${REPO_ROOT}/experiments/summarize_invariant_proxy_order_study.py" \
  --run-root "${RUN_ROOT}" --clean-root "${ORDER_CLEAN_ROOT}"
touch "${RUN_ROOT}/_EXECUTION_COMPLETE"
