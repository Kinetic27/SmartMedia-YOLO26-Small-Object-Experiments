#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-${VARIANT:-baseline}}"
DATA="${DATA:-$ROOT/configs/dataset.example.yaml}"
WEIGHTS="${WEIGHTS:-}"
IMGSZ="${IMGSZ:-1024}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
NAME="${NAME:-${VARIANT}_img${IMGSZ}}"
PATIENCE="${PATIENCE:-50}"
OPTIMIZER="${OPTIMIZER:-auto}"
LR0="${LR0:-0.01}"
EXTRA_ARGS=("$@")
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
else
  EXTRA_ARGS=()
fi

python "$ROOT/scripts/run_train.py"           --variant "$VARIANT"           --data "$DATA"           --weights "$WEIGHTS"           --imgsz "$IMGSZ"           --epochs "$EPOCHS"           --batch "$BATCH"           --device "$DEVICE"           --workers "$WORKERS"           --name "$NAME"           --patience "$PATIENCE"           --optimizer "$OPTIMIZER"           --lr0 "$LR0"           "${EXTRA_ARGS[@]}"
