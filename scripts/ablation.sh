#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-$ROOT/configs/dataset.example.yaml}"
IMGSZ="${IMGSZ:-1024}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"
WEIGHTS="${WEIGHTS:-}"

variants=(baseline p2 spd spd_p2 p2_nop5 spd_p2_nop5)
for variant in "${variants[@]}"; do
  echo "===== Running $variant ====="
  DATA="$DATA" IMGSZ="$IMGSZ" EPOCHS="$EPOCHS" BATCH="$BATCH" DEVICE="$DEVICE" WEIGHTS="$WEIGHTS"     bash "$ROOT/scripts/train.sh" "$variant"
done
