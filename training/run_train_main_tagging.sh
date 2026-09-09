#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
WEIGHT_PATH="${WEIGHT_PATH:-$ROOT_DIR/model}"
NPROC="${NPROC:-4}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/training_runs/data}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/training_runs/tagging_main}"
INIT_HEADS="${INIT_HEADS:-}"

mkdir -p "$OUT_DIR"

INIT_ARGS=()
if [[ -n "$INIT_HEADS" ]]; then
  INIT_ARGS=(--init-heads "$INIT_HEADS")
fi

"$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC" "$ROOT_DIR/training/train_tagging_heads.py" \
  --profile main \
  --train-jsonl "$DATA_DIR/tagging_train.jsonl" \
  --pseudo-jsonl "$DATA_DIR/quality_synthetic_train.jsonl" \
  --pseudo-jsonl "$DATA_DIR/event_caption_pseudo.jsonl" \
  --pseudo-weight 0.70 \
  --weight-path "$WEIGHT_PATH" \
  --vocab "$ROOT_DIR/model/task_a_vocab.json" \
  "${INIT_ARGS[@]}" \
  --out "$OUT_DIR/tagging_heads.pt" \
  --epochs "${TAGGING_EPOCHS:-10}" \
  --batch-size "${TAGGING_BATCH:-96}" \
  --lr "${TAGGING_LR:-1.0e-4}" \
  --warmup-ratio 0.08 \
  --optimizer adamw \
  --weight-decay 0.04 \
  --dropout 0.25 \
  --trunk-dim 768 \
  --focal-gamma 1.0 \
  --label-smoothing 0.01 \
  --quality-loss-weight 1.8 \
  --scene-loss-weight 1.0 \
  --event-loss-weight 1.7 \
  --synthetic-quality-prob 0.35 \
  --max-pos-weight 12.0 \
  --threshold-min 0.03 \
  --threshold-max 0.90 \
  --num-workers "${TAGGING_WORKERS:-6}"
