#!/usr/bin/env python3
"""Simple ViT inference helper for single-image prediction.

Usage example:
  python tools/vit_detect.py --image /path/to/img.png --dataset dataset/A

It will load a HuggingFace ViT model (default: google/vit-base-patch16-224-in21k),
attempt to load optional checkpoint weights, and print the predicted label
and confidence.
"""
from pathlib import Path
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTForImageClassification


def load_class_names_from_dataset(dataset_root: Path):
    if not dataset_root.exists():
        return None
    classes = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])
    return classes


def try_load_weights_into_model(model, ckpt_path: Path):
    """Attempt to load PyTorch checkpoint into the HF model with best-effort.
    Supports plain state_dict (.pth/.pt) or Lightning-style checkpoints (.ckpt).
    """
    try:
        data = torch.load(str(ckpt_path), map_location='cpu')
    except Exception as e:
        print('Failed to load checkpoint file:', e)
        return False

    # Lightning checkpoints wrap state dict under 'state_dict'
    if isinstance(data, dict) and 'state_dict' in data:
        sd = data['state_dict']
        # strip lightning prefix if present
        new_sd = {k.replace('vit.', '') if k.startswith('vit.') else k: v for k, v in sd.items()}
        try:
            model.load_state_dict(new_sd, strict=False)
            return True
        except Exception:
            # fallback to trying raw state dict
            try:
                model.load_state_dict(sd, strict=False)
                return True
            except Exception:
                return False

    if isinstance(data, dict):
        try:
            model.load_state_dict(data, strict=False)
            return True
        except Exception:
            return False

    return False


def predict_image(image_path: Path, hugging_model_name: str, checkpoint: Path = None, dataset_root: Path = None, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    print(f'Using device: {device}')

    feature_extractor = ViTFeatureExtractor.from_pretrained(hugging_model_name)
    model = ViTForImageClassification.from_pretrained(hugging_model_name)

    # Try to load class names from dataset folder if provided
    class_names = None
    if dataset_root:
        class_names = load_class_names_from_dataset(Path(dataset_root))

    # Try to load checkpoint weights if provided
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            ok = try_load_weights_into_model(model, ckpt_path)
            print(f'Weights load success: {ok}')
        else:
            print('Checkpoint not found:', ckpt_path)

    model.to(device)
    model.eval()

    img = Image.open(str(image_path)).convert('RGB')
    inputs = feature_extractor(images=img, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]
        conf, idx = torch.max(probs, dim=-1)
        conf = float(conf.cpu().item())
        idx = int(idx.cpu().item())

    label = None
    if class_names and idx < len(class_names):
        label = class_names[idx]
    else:
        # Use model's id2label if available
        id2label = getattr(model.config, 'id2label', None)
        if id2label and idx in id2label:
            label = id2label[idx]
        else:
            label = str(idx)

    return {'label': label, 'confidence': conf, 'index': idx}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='Path to image to classify')
    p.add_argument('--checkpoint', default=None, help='Optional checkpoint to load (.pth/.pt/.ckpt)')
    p.add_argument('--dataset', default=None, help='Optional dataset root to infer class names')
    p.add_argument('--model', default='google/vit-base-patch16-224-in21k', help='HuggingFace ViT model name')
    args = p.parse_args()

    res = predict_image(Path(args.image), args.model, checkpoint=Path(args.checkpoint) if args.checkpoint else None, dataset_root=Path(args.dataset) if args.dataset else None)
    print('Prediction:', res)


if __name__ == '__main__':
    main()
