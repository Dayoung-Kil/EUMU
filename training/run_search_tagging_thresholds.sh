#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
WEIGHT_PATH="${WEIGHT_PATH:-$ROOT_DIR/model}"
VAL_ROOT="${VAL_ROOT:-$ROOT_DIR/data/val_data}"
HEAD_IN="${HEAD_IN:-$ROOT_DIR/training_runs/final_tagging/tagging_heads.pt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/training_runs/threshold_search}"

mkdir -p "$OUT_DIR"

"$PYTHON" "$ROOT_DIR/training/search_tagging_thresholds.py" \
  --weight-path "$WEIGHT_PATH" \
  --head-in "$HEAD_IN" \
  --vocab "$ROOT_DIR/model/task_a_vocab.json" \
  --val-root "$VAL_ROOT" \
  --namespace scene \
  --subset scene \
  --max-labels 2 \
  --fallback-top-k 1 \
  --out "$OUT_DIR/scene_summary.json"

"$PYTHON" "$ROOT_DIR/training/search_tagging_thresholds.py" \
  --weight-path "$WEIGHT_PATH" \
  --head-in "$HEAD_IN" \
  --vocab "$ROOT_DIR/model/task_a_vocab.json" \
  --val-root "$VAL_ROOT" \
  --namespace event \
  --subset event \
  --max-labels 2 \
  --fallback-top-k 1 \
  --out "$OUT_DIR/event_summary.json"
