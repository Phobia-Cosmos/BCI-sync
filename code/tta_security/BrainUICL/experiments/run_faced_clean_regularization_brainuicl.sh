#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:-all}

PYTHON=/home/undefined/Disk/python-envs/brainuicl/bin/python
DATA=/home/undefined/Disk/datasets/FACED_processed
CHECKPOINTS=/home/undefined/Disk/ai-storage/BrainUICL/model_parameter
ROOT=${RUN_ROOT:-experiments/faced_clean_cl61_seed4321}
MAX_SUBJECTS=${MAX_SUBJECTS:-0}

mkdir -p "${ROOT}/logs"

run_regularization() {
  "${PYTHON}" experiments/regularization_cl_eeg.py \
    --dataset FACED \
    --data-root "${DATA}" \
    --input-checkpoint-root "${CHECKPOINTS}" \
    --output-root "${ROOT}/regularization" \
    --methods finetune,ewc,online_ewc,si,mas \
    --seed 4321 \
    --gpu 0 \
    --num-worker 0 \
    --max-subjects "${MAX_SUBJECTS}" \
    --ssl-epoch 10 \
    --incremental-epoch 10 \
    --lr 1e-4 \
    --ssl-lr 1e-6 \
    --cl-lr 1e-6 \
    --batch 16 \
    --beta1 0.5 \
    --beta2 0.99 \
    --weight-decay 3e-4 \
    --grad-clip 5 \
    --freeze-bn-stats \
    --ewc-strength 5000 \
    --online-ewc-strength 6500 \
    --online-ewc-decay 1 \
    --si-strength 1500000 \
    --si-xi 1e-6 \
    --mas-strength 3000 \
    --mas-decay 1 \
    --retention-milestones 10,25,61 \
    --checkpoint-milestones 0,1,10,25,61 \
    --no-save-checkpoints \
    2>&1 | tee "${ROOT}/logs/regularization.log"
}

run_brainuicl() {
  "${PYTHON}" experiments/rttdp_brainuicl_full.py \
    --dataset FACED \
    --data-root "${DATA}" \
    --input-checkpoint-root "${CHECKPOINTS}" \
    --output-root "${ROOT}/brainuicl" \
    --seed 4321 \
    --gpu 0 \
    --num-worker 0 \
    --max-subjects "${MAX_SUBJECTS}" \
    --batch 16 \
    --ssl-epoch 10 \
    --incremental-epoch 10 \
    --lr 1e-4 \
    --ssl-lr 1e-6 \
    --cl-lr 1e-6 \
    --beta1 0.5 \
    --beta2 0.99 \
    --weight-decay 3e-4 \
    --confidence 0.9 \
    --confident-epoch-n 15 \
    --freeze-bn-stats \
    --retention-milestones 10,25,61 \
    --checkpoint-milestones 0,1,10,25,61 \
    --no-save-checkpoints \
    --run-clean-only \
    2>&1 | tee "${ROOT}/logs/brainuicl.log"
}

case "${TARGET}" in
  regularization)
    run_regularization
    ;;
  brainuicl)
    run_brainuicl
    ;;
  all)
    run_regularization
    run_brainuicl
    ;;
  *)
    echo "usage: $0 [regularization|brainuicl|all]" >&2
    exit 2
    ;;
esac
