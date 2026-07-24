#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-experiments/canonical_n2n_shared_proxy/focused_ewc_plain_er_k5_q50_l2p20_steps5_seed4321}"
CLEAN_ROOT="${CLEAN_ROOT:-experiments/canonical_n2n_shared_proxy/full49_task26_q20_l2p20_seed4321}"
PAYLOAD_ROOT="$RUN_ROOT/shared_payload"
MANIFEST="$PAYLOAD_ROOT/manifest.json"
mkdir -p "$RUN_ROOT/logs"

run_logged() {
  local name="$1"
  shift
  printf '[run] %s %s\n' "$(date --iso-8601=seconds)" "$name"
  "$@" > "$RUN_ROOT/logs/$name.log" 2>&1
  printf '[done] %s %s\n' "$(date --iso-8601=seconds)" "$name"
}

if [[ ! -f "$MANIFEST" ]]; then
  run_logged generate_shared_payload \
    "$PYTHON" experiments/generate_n2n_shared_proxy_manifest.py \
      --output-root "$PAYLOAD_ROOT" \
      --affected-tasks 26,31,37,43,49 \
      --surrogate-resume-task 25 \
      --attack-fraction 0.50 \
      --attack-max-relative-l2 0.20 \
      --attack-eps-scale 0.50 \
      --attack-steps 5 \
      --attack-reference-batch 4 \
      --attack-generation-batch 4
else
  run_logged validate_shared_payload \
    "$PYTHON" experiments/generate_n2n_shared_proxy_manifest.py \
      --output-root "$PAYLOAD_ROOT" \
      --affected-tasks 26,31,37,43,49 \
      --surrogate-resume-task 25 \
      --attack-fraction 0.50 \
      --attack-max-relative-l2 0.20 \
      --attack-eps-scale 0.50 \
      --attack-steps 5
fi

EWC_OUTPUT="$RUN_ROOT/runs/regularization/ewc/shared_proxy"
if [[ ! -f "$EWC_OUTPUT/summary.json" ]]; then
  run_logged ewc_shared_proxy \
    "$PYTHON" experiments/regularization_cl_eeg.py \
      --methods ewc \
      --output-root "$EWC_OUTPUT" \
      --gpu 0 \
      --max-subjects 0 \
      --ssl-epoch 10 \
      --incremental-epoch 10 \
      --batch 16 \
      --num-worker 0 \
      --freeze-bn-stats \
      --no-save-checkpoints \
      --retention-milestones 10,25,49 \
      --checkpoint-milestones "" \
      --n2n-manifest "$MANIFEST" \
      --n2n-verify selected
else
  printf '[skip] EWC shared proxy\n'
fi

ER_OUTPUT="$RUN_ROOT/runs/replay/shared_proxy"
if [[ ! -f "$ER_OUTPUT/summary.json" ]]; then
  run_logged plain_er_shared_proxy \
    "$PYTHON" experiments/replay_cl_eeg.py \
      --method plain_er \
      --output-root "$ER_OUTPUT" \
      --gpu 0 \
      --max-subjects 0 \
      --ssl-epoch 10 \
      --incremental-epoch 10 \
      --batch 16 \
      --num-worker 0 \
      --freeze-bn-stats \
      --memory-capacity 1000 \
      --replay-ratio 1.0 \
      --retention-milestones 10,25,49 \
      --n2n-manifest "$MANIFEST" \
      --n2n-verify selected
else
  printf '[skip] Plain ER shared proxy\n'
fi

run_logged summarize \
  "$PYTHON" experiments/summarize_focused_multi_user_n2n.py \
    --run-root "$RUN_ROOT" \
    --clean-root "$CLEAN_ROOT" \
    --bci /home/undefined/Desktop/IPhone/BCI.md

run_logged tests \
  "$PYTHON" -m unittest discover -s tests -p 'test_*.py' -q

touch "$RUN_ROOT/_EXECUTION_COMPLETE"
printf '[complete] %s focused multi-user N-to-N validation\n' "$(date --iso-8601=seconds)"
