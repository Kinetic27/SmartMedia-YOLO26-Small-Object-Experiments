#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ASSETS_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0'
URLS = [
    f'{ASSETS_URL}/VisDrone2019-DET-train.zip',
    f'{ASSETS_URL}/VisDrone2019-DET-val.zip',
    f'{ASSETS_URL}/VisDrone2019-DET-test-dev.zip',
]
SPLITS = {
    'VisDrone2019-DET-train': 'train',
    'VisDrone2019-DET-val': 'val',
    'VisDrone2019-DET-test-dev': 'test',
}


def visdrone2yolo(root: Path, source_name: str, split: str) -> None:
    from PIL import Image
    try:
        from ultralytics.utils import TQDM
    except Exception:
        TQDM = lambda x, desc='': x  # noqa: E731

    source_dir = root / source_name
    images_dir = root / 'images' / split
    labels_dir = root / 'labels' / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    source_images = source_dir / 'images'
    if source_images.exists():
        for img in source_images.glob('*.jpg'):
            target = images_dir / img.name
            if not target.exists():
                img.rename(target)

    ann_dir = source_dir / 'annotations'
    for anno in TQDM(sorted(ann_dir.glob('*.txt')), desc=f'Converting {split}'):
        image_path = images_dir / f'{anno.stem}.jpg'
        if not image_path.exists():
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        dw, dh = 1.0 / width, 1.0 / height
        lines = []
        raw = anno.read_text(encoding='utf-8').strip()
        for row in raw.splitlines() if raw else []:
            parts = row.split(',')
            if len(parts) < 6:
                continue
            x, y, w, h = map(int, parts[:4])
            score = int(parts[4])
            cls_raw = int(parts[5])
            if score != 1 or not (1 <= cls_raw <= 10):
                continue
            cls = cls_raw - 1
            x_center = (x + w / 2) * dw
            y_center = (y + h / 2) * dh
            lines.append(f'{cls} {x_center:.6f} {y_center:.6f} {w * dw:.6f} {h * dh:.6f}\n')
        (labels_dir / anno.name).write_text(''.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Download and convert VisDrone2019-DET to YOLO format.')
    p.add_argument('--root', default='datasets/VisDrone')
    p.add_argument('--keep-archives', action='store_true')
    p.add_argument('--skip-download', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        from ultralytics.utils.downloads import download
        download(URLS, dir=root, threads=4)

    for folder, split in SPLITS.items():
        if not (root / folder).exists():
            raise FileNotFoundError(root / folder)
        visdrone2yolo(root, folder, split)
        shutil.rmtree(root / folder)

    if not args.keep_archives:
        for archive in root.glob('VisDrone2019-DET-*.zip'):
            archive.unlink(missing_ok=True)

    print(f'[done] VisDrone prepared at {root}')


if __name__ == '__main__':
    main()
