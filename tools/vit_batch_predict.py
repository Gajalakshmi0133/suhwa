#!/usr/bin/env python3
"""Batch-predict images in a folder using the ViT detector and write CSV output.

Usage:
  python -m tools.vit_batch_predict --input-dir path/to/images --output preds.csv

Supports optional `--checkpoint` and `--dataset` for class names.
"""
from pathlib import Path
import argparse
import csv
import sys
from typing import Iterable

from tools.vit_detect import predict_image


def iter_image_files(root: Path) -> Iterable[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() in exts:
            yield p


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', '-i', required=True, help='Directory or image file to predict')
    p.add_argument('--output', '-o', default='preds.csv', help='Output CSV file')
    p.add_argument('--checkpoint', '-c', default=None, help='Optional checkpoint to load')
    p.add_argument('--dataset', '-d', default=None, help='Optional dataset root to infer class names')
    p.add_argument('--model', default='google/vit-base-patch16-224-in21k', help='HuggingFace ViT model name')
    args = p.parse_args()

    inp = Path(args.input_dir)
    if not inp.exists():
        print('Input path does not exist:', inp)
        sys.exit(1)

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    ds = Path(args.dataset) if args.dataset else None

    files = list(iter_image_files(inp))
    if not files:
        print('No image files found under', inp)
        sys.exit(1)

    out_path = Path(args.output)
    with out_path.open('w', newline='', encoding='utf8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['image', 'label', 'confidence', 'index'])
        for f in files:
            try:
                res = predict_image(f, args.model, checkpoint=ckpt, dataset_root=ds)
                writer.writerow([str(f), res.get('label'), res.get('confidence'), res.get('index')])
                print(f'Predicted {f} -> {res.get("label")} ({res.get("confidence"):.3f})')
            except Exception as e:
                print('Error predicting', f, e)
                writer.writerow([str(f), None, 0.0, None])

    print('Wrote predictions to', out_path)


if __name__ == '__main__':
    main()
