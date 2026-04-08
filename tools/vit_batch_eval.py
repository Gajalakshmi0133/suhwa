#!/usr/bin/env python3
"""Evaluate ViT predictions on a dataset folder and write combined CSV with accuracy.

Usage:
  python -m tools.vit_batch_eval --dataset dataset/American_Sign_Language_Letters_Multiclass --output all_preds.csv
"""
from pathlib import Path
import argparse
import csv
import sys
from typing import Iterable, List

import torch
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import numpy as np


def list_classes(dataset_root: Path) -> List[str]:
    return sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])


def iter_image_files(root: Path) -> Iterable[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    for cls in sorted(root.iterdir()):
        if not cls.is_dir():
            continue
        for f in sorted(cls.rglob('*')):
            if f.suffix.lower() in exts:
                yield cls.name, f


def try_load_checkpoint(model, ckpt_path: Path):
    try:
        sd = torch.load(str(ckpt_path), map_location='cpu')
    except Exception as e:
        print('Failed to load checkpoint:', e)
        return False
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    # try to load permissively
    try:
        model.load_state_dict(sd, strict=False)
        return True
    except Exception:
        # try stripping common prefixes
        new_sd = {}
        for k, v in sd.items():
            nk = k
            for pfx in ('model.', 'vit.', 'module.', 'model.model.'):
                if nk.startswith(pfx):
                    nk = nk[len(pfx):]
            new_sd[nk] = v
        try:
            model.load_state_dict(new_sd, strict=False)
            return True
        except Exception as e:
            print('Failed to load state dict permissively:', e)
            return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Dataset root with class subfolders')
    p.add_argument('--output', default='all_preds.csv', help='CSV output file')
    p.add_argument('--checkpoint', default=None, help='Optional checkpoint to load')
    p.add_argument('--model', default='google/vit-base-patch16-224-in21k', help='HuggingFace ViT model name')
    args = p.parse_args()

    ds = Path(args.dataset)
    if not ds.exists() or not ds.is_dir():
        print('Dataset path invalid:', ds)
        sys.exit(1)

    classes = list_classes(ds)
    print('Found classes:', classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    fe = ViTFeatureExtractor.from_pretrained(args.model)
    model = ViTForImageClassification.from_pretrained(args.model)
    if args.checkpoint:
        ok = try_load_checkpoint(model, Path(args.checkpoint))
        print('Checkpoint load:', ok)
    model.to(device)
    model.eval()

    out_path = Path(args.output)
    total = 0
    correct = 0

    with out_path.open('w', newline='', encoding='utf8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['image', 'ground_truth', 'predicted', 'confidence', 'index'])

        for gt_class, img_path in iter_image_files(ds):
            try:
                img = Image.open(str(img_path)).convert('RGB')
                inputs = fe(images=img, return_tensors='pt')
                pixel_values = inputs['pixel_values'].to(device)
                with torch.no_grad():
                    outputs = model(pixel_values)
                    logits = outputs.logits.cpu().numpy()
                # softmax
                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp / np.sum(exp, axis=1, keepdims=True)
                prob = float(np.max(probs))
                idx = int(np.argmax(probs, axis=1)[0])
                # map index to class name if possible
                predicted = None
                if idx < len(classes):
                    predicted = classes[idx]
                else:
                    id2label = getattr(model.config, 'id2label', None)
                    if id2label and idx in id2label:
                        predicted = id2label[idx]
                    else:
                        predicted = f'LABEL_{idx}'

                writer.writerow([str(img_path), gt_class, predicted, f'{prob:.6f}', idx])
                total += 1
                if str(predicted).lower() == str(gt_class).lower():
                    correct += 1
            except Exception as e:
                print('Error processing', img_path, e)
                writer.writerow([str(img_path), gt_class, '', 0.0, ''])

    acc = (correct / total) if total else 0.0
    print(f'Wrote {out_path} — Total: {total}, Correct: {correct}, Accuracy: {acc:.4f}')


if __name__ == '__main__':
    main()
