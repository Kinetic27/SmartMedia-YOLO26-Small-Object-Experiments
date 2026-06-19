#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Validate a trained YOLO checkpoint.')
    p.add_argument('--model', required=True, help='path to trained .pt checkpoint')
    p.add_argument('--data', required=True, help='dataset yaml path')
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='0')
    p.add_argument('--project', default=str(ROOT / 'results' / 'runs'))
    p.add_argument('--name', default='val')
    p.add_argument('--dry-run', action='store_true', help='print validation arguments without importing Ultralytics')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f'[val] model={args.model}')
    if args.dry_run:
        print(f'data: {args.data}')
        print(f'imgsz: {args.imgsz}')
        print(f'batch: {args.batch}')
        print(f'device: {args.device}')
        print(f'project: {args.project}')
        print(f'name: {args.name}')
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
    )


if __name__ == '__main__':
    main()
