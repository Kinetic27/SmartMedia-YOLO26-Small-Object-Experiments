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
    p = argparse.ArgumentParser(description='Train a YOLO26 small-object experiment variant.')
    p.add_argument('--variant', choices=CONFIGS, default='baseline')
    p.add_argument('--data', required=True, help='dataset yaml path')
    p.add_argument('--weights', default='', help='optional pretrained checkpoint to load after model init')
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--project', default=str(ROOT / 'results' / 'runs'))
    p.add_argument('--name', default='')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--optimizer', default='auto')
    p.add_argument('--lr0', type=float, default=0.01)
    p.add_argument('--cos-lr', action='store_true')
    p.add_argument('--close-mosaic', type=int, default=10)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--cache', action='store_true')
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--exist-ok', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='print resolved config and arguments without importing Ultralytics')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = CONFIGS[args.variant]
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    run_name = args.name or f'{args.variant}_img{args.imgsz}'
    train_kwargs = {
        'data': args.data,
        'imgsz': args.imgsz,
        'epochs': args.epochs,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': run_name,
        'seed': args.seed,
        'patience': args.patience,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'close_mosaic': args.close_mosaic,
        'pretrained': bool(args.weights),
        'cache': args.cache,
        'cos_lr': args.cos_lr,
        'amp': args.amp,
        'plots': True,
        'fraction': args.fraction,
        'exist_ok': args.exist_ok,
    }
    print(f'[train] variant={args.variant} config={config_path}')
    print(f'[train] project={args.project} name={run_name}')
    if args.dry_run:
        for key, value in train_kwargs.items():
            print(f'{key}: {value}')
        return

    from ultralytics import YOLO

    model = YOLO(str(config_path))
    if args.weights:
        model = model.load(args.weights)
    model.train(**train_kwargs)


if __name__ == '__main__':
    main()
