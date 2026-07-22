#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}
RUN_ROOT=${RUN_ROOT:-"${REPO_ROOT}/experiments/regularization_cl_eeg_runs"}
CLEAN_ROOT=${CLEAN_ROOT:-"${RUN_ROOT}/clean49_bn_frozen_e10_lr1e6_seed4321"}
ATTACK_TASKS=1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49

if [[ ! -f "${CLEAN_ROOT}/summary.json" ]]; then
  "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
    --gpu 0 \
    --output-root "${CLEAN_ROOT}" \
    --methods ewc,online_ewc,si,mas \
    --max-subjects 0 \
    --ssl-epoch 10 \
    --incremental-epoch 10 \
    --batch 16 \
    --num-worker 0 \
    --freeze-bn-stats \
    --ewc-strength 5000 \
    --online-ewc-strength 6500 \
    --online-ewc-decay 1.0 \
    --si-strength 1500000 \
    --si-xi 1e-6 \
    --mas-strength 3000 \
    --mas-decay 1.0 \
    --no-save-checkpoints \
    --retention-milestones 10,25,49 \
    --checkpoint-milestones 10,25,49
fi

COMMON_ARGS=(
  --gpu 0
  --max-subjects 0
  --ssl-epoch 10
  --incremental-epoch 10
  --batch 16
  --num-worker 0
  --freeze-bn-stats
  --attack-mode proxy_dual_harm
  --attack-tasks "${ATTACK_TASKS}"
  --attack-fraction 1.0
  --attack-eps-scale 0.50
  --attack-max-relative-l2 0.20
  --attack-steps 3
  --attack-reference-batch 4
  --attack-generation-batch 4
  --attack-random-start
  --attack-target-weight 5.0
  --attack-conflict-weight 1.0
  --attack-gradient-norm-weight 0.25
  --attack-virtual-old-weight 1.0
  --attack-virtual-new-weight 1.0
  --attack-new-proxy-weight 1.0
  --attack-curvature-scale 1.0
  --attack-min-confidence 0.85
  --attack-confidence-weight 2.0
  --attack-l2-weight 0.01
  --attack-proxy-repeat 3
  --delete-poisoned-inputs-after-task
  --no-save-checkpoints
  --retention-milestones 10,25,49
  --checkpoint-milestones 10,25,49
)

run_method() {
  local method=$1
  local directory=$2
  shift 2
  "${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
    --output-root "${RUN_ROOT}/${directory}" \
    --methods "${method}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

run_method ewc proxy_dual_harm_repeat3_odd25_full49_e10 \
  --ewc-strength 5000
run_method online_ewc proxy_dual_harm_repeat3_odd25_online_ewc_full49_e10 \
  --online-ewc-strength 6500 --online-ewc-decay 1.0
run_method si proxy_dual_harm_repeat3_odd25_si_full49_e10 \
  --si-strength 1500000 --si-xi 1e-6
run_method mas proxy_dual_harm_repeat3_odd25_mas_full49_e10 \
  --mas-strength 3000 --mas-decay 1.0

"${PYTHON}" "${REPO_ROOT}/experiments/regularization_cl_eeg.py" \
  --gpu 0 \
  --output-root "${RUN_ROOT}/benign_repeat3_odd25_full49_e10" \
  --methods ewc,online_ewc,si,mas \
  --max-subjects 0 \
  --ssl-epoch 10 \
  --incremental-epoch 10 \
  --batch 16 \
  --num-worker 0 \
  --freeze-bn-stats \
  --ewc-strength 5000 \
  --online-ewc-strength 6500 \
  --online-ewc-decay 1.0 \
  --si-strength 1500000 \
  --si-xi 1e-6 \
  --mas-strength 3000 \
  --mas-decay 1.0 \
  --attack-mode benign_repeat \
  --attack-tasks "${ATTACK_TASKS}" \
  --attack-proxy-repeat 3 \
  --delete-poisoned-inputs-after-task \
  --no-save-checkpoints \
  --retention-milestones 10,25,49 \
  --checkpoint-milestones 10,25,49

"${PYTHON}" "${REPO_ROOT}/experiments/summarize_proxy_dual_harm.py" \
  --run-root "${RUN_ROOT}" \
  --clean-root "${CLEAN_ROOT}" \
  --benign-root "${RUN_ROOT}/benign_repeat3_odd25_full49_e10"
