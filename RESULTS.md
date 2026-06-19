# Results and Fairness Notes

This repository keeps public-safe metrics and reproducibility notes only. Raw logs, generated validation dumps, checkpoints, figures, and dataset files are excluded.

## Official YOLO26n checkpoint follow-up

A later check inspected official `yolo26n.pt` metadata. The checkpoint recorded mAP50-95 `0.40116` and mAP50 `0.55715`, but its metadata pointed to an intermediate pretrained checkpoint and internal training knobs rather than a plain public scratch command. Therefore, exact official checkpoint reproduction is not claimed.

Observed local official-family scratch runs from the separate official checkout included:

| Run | Model | Epochs | Batch | mAP50 | mAP50-95 | Note |
|---|---|---:|---:|---:|---:|---|
| official scratch | `yolo26n.yaml` | 100 | 16 | 0.49258 | 0.34727 | public scratch recipe, lower fidelity than released checkpoint |
| official scratch | `yolo26n.yaml` | 300 | 64 | 0.52779 | 0.37420 | longer public scratch recipe, still below released checkpoint |
| released checkpoint metadata | `yolo26n.pt` | 245 | 128 | 0.55715 | 0.40116 | uses intermediate pretrained source/internal knobs |

## Variant set retained for public ablation

| Variant | Public config basis | Change |
|---|---|---|
| `baseline` | official `yolo26.yaml` | none |
| `p2` | official `yolo26-p2.yaml` | none |
| `spd` | official `yolo26.yaml` | first two downsampling layers changed to `Focus` |
| `spd_p2` | official `yolo26-p2.yaml` | first two downsampling layers changed to `Focus` |
| `p2_nop5` | official `yolo26-p2.yaml` | P5 detection head removed |
| `spd_p2_nop5` | official `yolo26-p2.yaml` | SPD-style stem plus P5 detection head removed |

## Included metric files

- `results/metrics.csv`: quick sanity-run metrics and latency notes only.
- `results/visdrone_small_object_stats.json`: VisDrone label-size distribution summary.

## VisDrone small-object distribution

The prepared VisDrone training split contained 343,205 boxes. About 97% were very-small or small by normalized area, making it a suitable stress test for high-resolution detection-head experiments.
