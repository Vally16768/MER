#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-baseline_mfcc80_bs128}"
NUM_WORKERS="${NUM_WORKERS:-16}"
N_MFCC="${N_MFCC:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="./.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi

if [[ -d "outputs/${RUN_NAME}" ]]; then
  echo "Run dir already exists: outputs/${RUN_NAME}. Pick a new RUN_NAME or delete the folder to avoid appending logs."
  exit 1
fi

need_splits=(train val testA testB)
missing=0
for s in "${need_splits[@]}"; do
  if [[ ! -d "data/features/mer_builder_at_simple/${s}" ]]; then
    missing=1
  fi
done

if [[ "$missing" -eq 1 ]]; then
  echo "Missing feature splits; running feature extraction..."
  "$PY" extract_at_features.py \
    --processed_dir mer_dataset_builder/data/processed \
    --out_dir data/features/mer_builder_at_simple \
    --n_mfcc "$N_MFCC" \
    --num_workers "$NUM_WORKERS"
fi

"$PY" train.py \
  --config configs/mer_builder_at_simple.yaml \
  --modalities A T \
  --run_name "$RUN_NAME" \
  --set "training.batch_size=${BATCH_SIZE}"

"$PY" evaluate.py \
  --config "outputs/${RUN_NAME}/config_resolved.yaml" \
  --ckpt "outputs/${RUN_NAME}/checkpoints/best.pt" \
  --set "data.eval_dir=data/features/mer_builder_at_simple/testA" \
  --output_dir "outputs/${RUN_NAME}/eval_testA"

"$PY" evaluate.py \
  --config "outputs/${RUN_NAME}/config_resolved.yaml" \
  --ckpt "outputs/${RUN_NAME}/checkpoints/best.pt" \
  --set "data.eval_dir=data/features/mer_builder_at_simple/testB" \
  --output_dir "outputs/${RUN_NAME}/eval_testB"

echo "Done. Outputs: outputs/${RUN_NAME}"

