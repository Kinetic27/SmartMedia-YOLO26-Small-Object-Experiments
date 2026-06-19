#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

VERY_SMALL_MAX = 0.001
SMALL_MAX = 0.009
MEDIUM_MAX = 0.04


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Summarize YOLO label box-size distribution using normalized box area.')
    p.add_argument('--labels-dir', required=True, help='directory containing YOLO txt labels')
    p.add_argument('--output', default='', help='optional JSON output path')
    return p.parse_args()


def classify(area_ratio: float) -> str:
    if area_ratio < VERY_SMALL_MAX:
        return 'very_small'
    if area_ratio < SMALL_MAX:
        return 'small'
    if area_ratio < MEDIUM_MAX:
        return 'medium'
    return 'large'


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)

    counts = {'very_small': 0, 'small': 0, 'medium': 0, 'large': 0}
    total_boxes = 0
    total_files = 0

    for path in sorted(labels_dir.rglob('*.txt')):
        total_files += 1
        for line in path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, _, _, w, h, *_ = parts
            area_ratio = float(w) * float(h)
            counts[classify(area_ratio)] += 1
            total_boxes += 1

    summary = {
        'labels_dir': str(labels_dir),
        'files': total_files,
        'boxes': total_boxes,
        'counts': counts,
        'ratios': {k: (v / total_boxes if total_boxes else 0.0) for k, v in counts.items()},
        'recommendation': (
            'P2 + SPD 우선 검토'
            if total_boxes and (counts['very_small'] + counts['small']) / total_boxes >= 0.5
            else 'P2 우선, SPD는 선택적 검토'
        ),
        'thresholds': {
            'very_small_max_area_ratio': VERY_SMALL_MAX,
            'small_max_area_ratio': SMALL_MAX,
            'medium_max_area_ratio': MEDIUM_MAX,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
