# SmartMedia YOLO26 Small-Object Experiments

Public, reproducible YOLO26-family small-object detection ablation code used for SmartMedia experiments. This repository keeps only the fair public surface: official YOLO26/YOLO26-P2 configs, single-axis derived variants, dataset-preparation utilities, training/validation scripts, and public-safe notes.

## Fairness boundary

The public repo intentionally excludes legacy exploratory baselines that were not based on the official YOLO26 family. The included variants are derived from official Ultralytics YOLO26 config files in a controlled way:

- `baseline`: official `yolo26.yaml`.
- `p2`: official `yolo26-p2.yaml`.
- `spd`: official `yolo26.yaml` with only the first two downsampling layers changed from `Conv` to `Focus` as a space-to-depth-style stem.
- `spd_p2`: official `yolo26-p2.yaml` with the same SPD-style stem change.
- `p2_nop5`: official `yolo26-p2.yaml` with the P5 detection head removed.
- `spd_p2_nop5`: combines the SPD-style stem and P5-head removal changes.

This makes the comparison fair as a controlled architecture ablation. Exact reproduction of the released official `yolo26n.pt` checkpoint is not claimed, because the checkpoint metadata points to an intermediate pretrained model and internal training knobs. See `RESULTS.md` for the reproduction boundary.

## Repository layout

```text
configs/   Official YOLO26 configs and controlled derived variants
scripts/   data conversion, training, validation, ablation, and analysis utilities
results/   public-safe sanity metrics and dataset statistics
```

Raw datasets, checkpoints, generated runs, logs, paper drafts, slides, and private environment files are intentionally excluded.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use a PyTorch/CUDA build that matches your machine if the default resolver is not suitable.

## Prepare VisDrone

```bash
python scripts/prepare_visdrone.py --root datasets/VisDrone
python scripts/analyze_small_objects.py --labels-dir datasets/VisDrone/labels/train --output results/visdrone_small_object_stats.json
```

The conversion keeps VisDrone boxes with `score == 1` and `object_category` in `1..10`, then remaps categories to zero-based YOLO labels.

## Run experiments

Quick static argument check without loading Ultralytics:

```bash
python scripts/run_train.py --variant baseline --data configs/visdrone.yaml --dry-run
```

Single variant training:

```bash
DATA=configs/visdrone.yaml IMGSZ=960 EPOCHS=50 BATCH=8 DEVICE=0 bash scripts/train.sh baseline
DATA=configs/visdrone.yaml IMGSZ=960 EPOCHS=50 BATCH=4 DEVICE=0 bash scripts/train.sh p2
```

Sequential ablation:

```bash
DATA=configs/visdrone.yaml IMGSZ=960 EPOCHS=50 bash scripts/ablation.sh
```

Official-checkpoint public recipe approximation:

```bash
DRY_RUN=1 DATA=configs/coco2017.example.yaml bash scripts/run_official_yolo26n_recipe_approx.sh
```

## Results

See `RESULTS.md` for public-safe metrics and the official-checkpoint reproduction boundary.

## License

AGPL-3.0-only. This project uses Ultralytics YOLO tooling, which is available under AGPL-3.0 or a commercial Ultralytics Enterprise license.
