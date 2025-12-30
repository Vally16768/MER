#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RAW_DIR="${RAW_DIR:-data/raw}"
OUT_DIR="${OUT_DIR:-data/processed}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MEAD_CONTEMPT="${MEAD_CONTEMPT:-drop}"

python -m pip install -e .
python -m mer_builder all \
  --raw_dir "$RAW_DIR" \
  --out_dir "$OUT_DIR" \
  --num_workers "$NUM_WORKERS" \
  --mead_contempt "$MEAD_CONTEMPT"
