#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
MAIN_HEAD="${MAIN_HEAD:-$ROOT_DIR/training_runs/tagging_main/tagging_heads.pt}"
QUALITY_HEAD="${QUALITY_HEAD:-$ROOT_DIR/training_runs/tagging_quality/tagging_heads.pt}"
SCENE_SUMMARY="${SCENE_SUMMARY:-}"
EVENT_SUMMARY="${EVENT_SUMMARY:-}"
OUT="${OUT:-$ROOT_DIR/training_runs/final_tagging/tagging_heads.pt}"

SUMMARY_ARGS=()
if [[ -n "$SCENE_SUMMARY" ]]; then
  SUMMARY_ARGS+=(--scene-summary "$SCENE_SUMMARY")
fi
if [[ -n "$EVENT_SUMMARY" ]]; then
  SUMMARY_ARGS+=(--event-summary "$EVENT_SUMMARY")
fi

"$PYTHON" "$ROOT_DIR/training/assemble_tagging_heads.py" \
  --main-head "$MAIN_HEAD" \
  --quality-head "$QUALITY_HEAD" \
  --vocab "$ROOT_DIR/model/task_a_vocab.json" \
  "${SUMMARY_ARGS[@]}" \
  --out "$OUT" \
  "$@"
