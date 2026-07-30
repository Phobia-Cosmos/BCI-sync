#!/usr/bin/env bash
set -euo pipefail

METHOD=${1:?usage: run_faced_full_spr_puridiver_proxy.sh full_spr|full_puridiver static|feedback}
MODE=${2:?usage: run_faced_full_spr_puridiver_proxy.sh full_spr|full_puridiver static|feedback}

PYTHON=/home/undefined/Disk/python-envs/brainuicl/bin/python
DATA=/home/undefined/Disk/datasets/FACED_processed
CHECKPOINTS=/home/undefined/Disk/ai-storage/BrainUICL/model_parameter
ROOT=${RUN_ROOT:-experiments/faced_aligned_full61_seed4321}
MAX_SUBJECTS=${MAX_SUBJECTS:-0}
OUT=${ROOT}/${METHOD}_${MODE}

PROXY_COMMON=(
  --dataset FACED
  --data-root "${DATA}"
  --input-checkpoint-root "${CHECKPOINTS}"
  --output-root "${OUT}"
  --seed 4321
  --gpu 0
  --num-worker 0
  --max-subjects "${MAX_SUBJECTS}"
  --ssl-epoch 10
  --lr 1e-4
  --ssl-lr 1e-6
  --beta1 0.5
  --beta2 0.99
  --weight-decay 3e-4
  --grad-clip 5
  --retention-milestones 10,25,61
  --no-save-state
  --progressive-proxy-mode "${MODE}"
  --progressive-proxy-tasks odd
  --progressive-clean-feedback-tasks even
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
  --progressive-upload-full-pool
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
  --progressive-require-source-conflict
)

mkdir -p "${ROOT}/logs"
if [[ "${METHOD}" == "full_spr" ]]; then
  "${PYTHON}" experiments/full_spr_eeg_adapted.py \
    "${PROXY_COMMON[@]}" \
    --batch 8 \
    --eval-batch 32 \
    --expert-epochs 10 \
    --base-epochs 10 \
    --ft-epochs 10 \
    --spr-ssl-lr 1e-6 \
    --ft-lr 1e-6 \
    --delayed-capacity-sequences 32 \
    --memory-capacity-epochs 1000 \
    --freeze-spr-bn-stats \
    2>&1 | tee "${ROOT}/logs/${METHOD}_${MODE}.log"
elif [[ "${METHOD}" == "full_puridiver" ]]; then
  "${PYTHON}" experiments/full_puridiver_eeg_adapted.py \
    "${PROXY_COMMON[@]}" \
    --batch 8 \
    --online-batch-sequences 8 \
    --replay-batch-sequences 8 \
    --infer-batch 16 \
    --eval-batch 32 \
    --cl-lr 1e-6 \
    --memory-capacity-epochs 1000 \
    --replay-epochs 10 \
    --warmup-epochs 2 \
    --freeze-student-bn-stats \
    2>&1 | tee "${ROOT}/logs/${METHOD}_${MODE}.log"
else
  echo "unknown method: ${METHOD}" >&2
  exit 2
fi
