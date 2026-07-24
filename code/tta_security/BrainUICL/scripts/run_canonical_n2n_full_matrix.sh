#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/undefined/Disk/python-envs/brainuicl/bin/python}"
RUN_ROOT="${RUN_ROOT:-experiments/canonical_n2n_shared_proxy/full49_task26_q20_l2p20_seed4321}"
PAYLOAD_ROOT="$RUN_ROOT/shared_payload"
MANIFEST="$PAYLOAD_ROOT/manifest.json"
mkdir -p "$RUN_ROOT/logs"

summary_has() {
  local method="$1"
  local path="$2"
  [[ -f "$path" ]] && jq -e --arg method "$method" 'has($method)' "$path" >/dev/null
}

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
      --affected-task 26 \
      --surrogate-resume-task 25 \
      --attack-fraction 0.20 \
      --attack-max-relative-l2 0.20 \
      --attack-eps-scale 0.50 \
      --attack-steps 3 \
      --attack-reference-batch 4 \
      --attack-generation-batch 4
else
  run_logged validate_shared_payload \
    "$PYTHON" experiments/generate_n2n_shared_proxy_manifest.py \
      --output-root "$PAYLOAD_ROOT"
fi

COMMON_REG=(
  --gpu 0
  --max-subjects 0
  --ssl-epoch 10
  --incremental-epoch 10
  --batch 16
  --num-worker 0
  --freeze-bn-stats
  --no-save-checkpoints
  --retention-milestones 10,25,49
  --checkpoint-milestones ""
)

run_regularization() {
  local method="$1"
  local condition="$2"
  local output="$RUN_ROOT/runs/regularization/$method/$condition"
  if summary_has "$method" "$output/summary.json"; then
    printf '[skip] regularization/%s/%s\n' "$method" "$condition"
    return
  fi
  local extra=()
  if [[ "$condition" == "shared_proxy" ]]; then
    extra=(--n2n-manifest "$MANIFEST" --n2n-verify selected)
  fi
  run_logged "regularization_${method}_${condition}" \
    "$PYTHON" experiments/regularization_cl_eeg.py \
      --methods "$method" \
      --output-root "$output" \
      "${COMMON_REG[@]}" \
      "${extra[@]}"
}

for method in finetune ewc online_ewc si mas; do
  run_regularization "$method" clean
  run_regularization "$method" shared_proxy
done

COMMON_REPLAY=(
  --method plain_er
  --gpu 0
  --max-subjects 0
  --ssl-epoch 10
  --incremental-epoch 10
  --batch 16
  --num-worker 0
  --freeze-bn-stats
  --memory-capacity 1000
  --replay-ratio 1.0
  --retention-milestones 10,25,49
)

for condition in clean shared_proxy; do
  output="$RUN_ROOT/runs/replay/$condition"
  if summary_has plain_er "$output/summary.json"; then
    printf '[skip] replay/plain_er/%s\n' "$condition"
    continue
  fi
  extra=()
  if [[ "$condition" == "shared_proxy" ]]; then
    extra=(--n2n-manifest "$MANIFEST" --n2n-verify selected)
  fi
  run_logged "replay_plain_er_${condition}" \
    "$PYTHON" experiments/replay_cl_eeg.py \
      --output-root "$output" \
      "${COMMON_REPLAY[@]}" \
      "${extra[@]}"
done

for condition in clean shared_proxy; do
  output="$RUN_ROOT/runs/full_spr/$condition"
  if summary_has full_spr_eeg_adapted "$output/summary.json"; then
    printf '[skip] full_spr/%s\n' "$condition"
    continue
  fi
  extra=()
  if [[ "$condition" == "shared_proxy" ]]; then
    extra=(--n2n-manifest "$MANIFEST" --n2n-verify selected)
  fi
  run_logged "full_spr_${condition}" \
    "$PYTHON" experiments/full_spr_eeg_adapted.py \
      --output-root "$output" \
      --gpu 0 \
      --max-subjects 0 \
      --ssl-epoch 10 \
      --expert-epochs 10 \
      --base-epochs 10 \
      --ft-epochs 10 \
      --batch 8 \
      --eval-batch 32 \
      --num-worker 0 \
      --delayed-capacity-sequences 32 \
      --memory-capacity-epochs 1000 \
      --freeze-spr-bn-stats \
      --retention-milestones 10,25,49 \
      --no-save-state \
      "${extra[@]}"
done

for condition in clean shared_proxy; do
  output="$RUN_ROOT/runs/full_puridiver/$condition"
  if summary_has full_puridiver_eeg_adapted "$output/summary.json"; then
    printf '[skip] full_puridiver/%s\n' "$condition"
    continue
  fi
  extra=()
  if [[ "$condition" == "shared_proxy" ]]; then
    extra=(--n2n-manifest "$MANIFEST" --n2n-verify selected)
  fi
  run_logged "full_puridiver_${condition}" \
    "$PYTHON" experiments/full_puridiver_eeg_adapted.py \
      --output-root "$output" \
      --gpu 0 \
      --max-subjects 0 \
      --ssl-epoch 10 \
      --online-batch-sequences 8 \
      --replay-batch-sequences 8 \
      --infer-batch 16 \
      --eval-batch 32 \
      --num-worker 0 \
      --memory-capacity-epochs 1000 \
      --replay-epochs 10 \
      --warmup-epochs 2 \
      --freeze-student-bn-stats \
      --retention-milestones 10,25,49 \
      --no-save-state \
      "${extra[@]}"
done

run_logged summarize \
  "$PYTHON" experiments/summarize_canonical_n2n_matrix.py \
    --run-root "$RUN_ROOT" \
    --bci /home/undefined/Desktop/IPhone/BCI.md

run_logged tests \
  "$PYTHON" -m unittest discover -s tests -p 'test_*.py' -q

touch "$RUN_ROOT/_EXECUTION_COMPLETE"
printf '[complete] %s canonical N-to-N full matrix\n' "$(date --iso-8601=seconds)"
