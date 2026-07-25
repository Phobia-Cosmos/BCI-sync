#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-experiments/canonical_n2n_shared_proxy/dose_sweep_ewc_plain_er_nested_seed4321}"
CLEAN_ROOT="${CLEAN_ROOT:-experiments/canonical_n2n_shared_proxy/full49_task26_q20_l2p20_seed4321}"
MAX_PAYLOAD="$RUN_ROOT/max_payload"
MAX_MANIFEST="$MAX_PAYLOAD/manifest.json"
ALL_TASKS="1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49"
STAGES=(k01_q20 k05_q20 k05_q50 k10_q50 k25_q50 k25_q100)
mkdir -p "$RUN_ROOT/logs"

run_logged() {
  local name="$1"
  shift
  printf '[run] %s %s\n' "$(date --iso-8601=seconds)" "$name"
  "$@" > "$RUN_ROOT/logs/$name.log" 2>&1
  printf '[done] %s %s\n' "$(date --iso-8601=seconds)" "$name"
}

if [[ ! -f "$MAX_MANIFEST" ]]; then
  run_logged generate_max_payload \
    "$PYTHON" experiments/generate_n2n_shared_proxy_manifest.py \
      --output-root "$MAX_PAYLOAD" \
      --surrogate-resume-task 0 \
      --affected-tasks "$ALL_TASKS" \
      --attack-fraction 1.00 \
      --attack-max-relative-l2 0.20 \
      --attack-eps-scale 0.50 \
      --attack-steps 5 \
      --attack-reference-batch 4 \
      --attack-generation-batch 4
else
  run_logged validate_max_payload \
    "$PYTHON" experiments/generate_n2n_shared_proxy_manifest.py \
      --output-root "$MAX_PAYLOAD" \
      --surrogate-resume-task 0 \
      --affected-tasks "$ALL_TASKS" \
      --attack-fraction 1.00 \
      --attack-max-relative-l2 0.20 \
      --attack-eps-scale 0.50 \
      --attack-steps 5
fi

run_logged derive_nested_manifests \
  "$PYTHON" experiments/derive_nested_n2n_sweep.py \
    --max-manifest "$MAX_MANIFEST" \
    --output-root "$RUN_ROOT"

for stage in "${STAGES[@]}"; do
  manifest="$RUN_ROOT/stages/$stage/manifest.json"
  ewc_output="$RUN_ROOT/stages/$stage/runs/regularization/ewc/shared_proxy"
  if [[ ! -f "$ewc_output/summary.json" ]]; then
    run_logged "${stage}_ewc" \
      "$PYTHON" experiments/regularization_cl_eeg.py \
        --methods ewc \
        --output-root "$ewc_output" \
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
        --n2n-manifest "$manifest" \
        --n2n-verify selected
  else
    printf '[skip] %s EWC\n' "$stage"
  fi

  er_output="$RUN_ROOT/stages/$stage/runs/replay/shared_proxy"
  if [[ ! -f "$er_output/summary.json" ]]; then
    run_logged "${stage}_plain_er" \
      "$PYTHON" experiments/replay_cl_eeg.py \
        --method plain_er \
        --output-root "$er_output" \
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
        --n2n-manifest "$manifest" \
        --n2n-verify selected
  else
    printf '[skip] %s Plain ER\n' "$stage"
  fi
done

run_logged summarize \
  "$PYTHON" experiments/summarize_nested_n2n_dose_sweep.py \
    --run-root "$RUN_ROOT" \
    --clean-root "$CLEAN_ROOT" \
    --bci /home/undefined/Desktop/IPhone/BCI.md

run_logged tests \
  "$PYTHON" -m unittest discover -s tests -p 'test_*.py' -q

touch "$RUN_ROOT/_EXECUTION_COMPLETE"
printf '[complete] %s nested N-to-N dose sweep\n' "$(date --iso-8601=seconds)"
