#!/usr/bin/env python3
"""Demo: run a single image through the ViT detector and print result.

Finds a sample image inside the dataset root (first class, first image)
unless an explicit `--image` is provided.
"""
from pathlib import Path
import argparse
import sys

from tools.vit_detect import predict_image


def find_sample_image(dataset_root: Path):
    if not dataset_root.exists():
        return None
    for cls in sorted(dataset_root.iterdir()):
        if cls.is_dir():
            for f in sorted(cls.iterdir()):
                if f.is_file():
                    return f
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='dataset/A', help='Dataset root to pick a sample image')
    p.add_argument('--image', default=None, help='Explicit image path to predict')
    p.add_argument('--checkpoint', default=None, help='Optional checkpoint to load')
    p.add_argument('--model', default='google/vit-base-patch16-224-in21k', help='HuggingFace ViT model')
    args = p.parse_args()

    image_path = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print('Image not found:', image_path)
            sys.exit(1)
    else:
        ds = Path(args.dataset)
        image_path = find_sample_image(ds)
        if image_path is None:
            print('No sample image found in dataset. Provide --image instead.')
            sys.exit(1)

    print('Using image:', image_path)
    res = predict_image(image_path, args.model, checkpoint=Path(args.checkpoint) if args.checkpoint else None, dataset_root=Path(args.dataset))
    print('Prediction result:', res)


if __name__ == '__main__':
    main()
