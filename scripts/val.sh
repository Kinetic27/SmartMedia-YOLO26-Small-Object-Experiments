#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-${1:-}}"
if [ -z "$MODEL" ]; then
  echo "Usage: MODEL=/path/to/best.pt bash scripts/val.sh [optional-model-path]" >&2
  exit 1
fi
DATA="${DATA:-$ROOT/configs/dataset.example.yaml}"
IMGSZ="${IMGSZ:-1024}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"
NAME="${NAME:-val}"

python "$ROOT/scripts/run_val.py"           --model "$MODEL"           --data "$DATA"           --imgsz "$IMGSZ"           --batch "$BATCH"           --device "$DEVICE"           --name "$NAME"
