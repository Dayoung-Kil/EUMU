#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SOURCE_DIR="${SOURCE_DIR:-$ROOT_DIR/data/train_data}"
CAPTION_JSONL="${CAPTION_JSONL:-$SOURCE_DIR/train_caption.jsonl}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/training_runs/data}"
IMAGE_ROOTS="${IMAGE_ROOTS:-}"

IMAGE_ROOT_ARGS=()
if [[ -n "$IMAGE_ROOTS" ]]; then
  IFS=':' read -r -a ROOTS <<< "$IMAGE_ROOTS"
  for root in "${ROOTS[@]}"; do
    IMAGE_ROOT_ARGS+=(--image-root "$root")
  done
fi

"$PYTHON" "$ROOT_DIR/training/prepare_tagging_data.py" \
  --source-dir "$SOURCE_DIR" \
  --caption-jsonl "$CAPTION_JSONL" \
  --out-dir "$OUT_DIR" \
  "${IMAGE_ROOT_ARGS[@]}"
