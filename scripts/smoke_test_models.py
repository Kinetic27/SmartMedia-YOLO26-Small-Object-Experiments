#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'baseline': ROOT / 'configs' / 'yolo26.yaml',
    'p2': ROOT / 'configs' / 'yolo26-p2.yaml',
    'spd': ROOT / 'configs' / 'yolo26-spd.yaml',
    'spd_p2': ROOT / 'configs' / 'yolo26-spd-p2.yaml',
    'p2_nop5': ROOT / 'configs' / 'yolo26-p2-nop5.yaml',
    'spd_p2_nop5': ROOT / 'configs' / 'yolo26-spd-p2-nop5.yaml',
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Instantiate custom configs and run a dummy forward pass.')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from ultralytics import YOLO

    x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=args.device)
    for name, config in CONFIGS.items():
        model = YOLO(str(config)).model.to(args.device)
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        if isinstance(outputs, (list, tuple)):
            summary = [tuple(o.shape) for o in outputs if hasattr(o, 'shape')]
        else:
            summary = getattr(outputs, 'shape', type(outputs).__name__)
        print(f'[ok] {name}: {summary}')


if __name__ == '__main__':
    main()
