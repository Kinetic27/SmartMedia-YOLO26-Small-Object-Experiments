#!/usr/bin/env bash
set -euo pipefail

# Public-code approximation of the official YOLO26n COCO training recipe.
# Exact reproduction of the released checkpoint is not guaranteed because the
# checkpoint metadata points to an intermediate pretrained model and internal
# training knobs that may not be exposed by public CLI arguments.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-$ROOT/configs/coco2017.example.yaml}"
PROJECT="${PROJECT:-$ROOT/results/runs}"
NAME="${NAME:-official_yolo26n_public_recipe_approx}"
DEVICE="${DEVICE:-0,1}"
BATCH="${BATCH:-128}"
MODEL="${MODEL:-yolo26n.yaml}"
PRETRAINED="${PRETRAINED:-False}"
DRY_RUN="${DRY_RUN:-0}"

ARGS=(
  detect train
  model="$MODEL"
  data="$DATA"
  epochs=245
  batch="$BATCH"
  imgsz=640
  device="$DEVICE"
  workers=8
  optimizer=MuSGD
  lr0=0.0054
  lrf=0.04952
  momentum=0.94676
  weight_decay=0.00064
  warmup_epochs=0.98124
  warmup_momentum=0.6576
  warmup_bias_lr=0.08114
  box=5.62767
  cls=0.56099
  dfl=9.03871
  mosaic=0.90863
  mixup=0.01216
  copy_paste=0.07504
  scale=0.56232
  fliplr=0.60571
  flipud=0.05854
  degrees=1.11032
  shear=1.46386
  translate=0.07105
  hsv_h=0.01373
  hsv_s=0.64481
  hsv_v=0.56565
  bgr=0.10567
  perspective=0.00011
  close_mosaic=10
  end2end=True
  pretrained="$PRETRAINED"
  save_json=True
  project="$PROJECT"
  name="$NAME"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf "yolo"
  printf " %q" "${ARGS[@]}"
  printf "\n"
  exit 0
fi

yolo "${ARGS[@]}"
