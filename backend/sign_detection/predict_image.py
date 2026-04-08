import io
import os
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from .model_loader import load_model


def _default_class_names():
    import string
    names = list(string.ascii_lowercase[:26])
    names.remove('j')
    names.remove('z')
    return names


def load_image_model(model_path: str):
    """Return (model, class_names). class_names is list[str]."""
    model = load_model(model_path)
    class_names = None
    if model_path:
        labels_path = os.path.join(os.path.dirname(model_path), 'labels.txt')
        if os.path.exists(labels_path):
            with open(labels_path, encoding='utf8') as f:
                class_names = [l.strip() for l in f.read().splitlines() if l.strip()]
    if class_names is None:
        class_names = getattr(model, 'class_names', None) or _default_class_names()
    return model, class_names


def _preprocess_pil_image(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB')
    img = img.resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def _preprocess_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    arr = small.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_from_image_bytes(image_bytes: bytes, model_path: Optional[str] = None, model=None, class_names=None) -> dict:
    """Predict from image bytes.
    Accepts either a preloaded `model` (+ optional `class_names`) or a `model_path` to load.
    """
    img = Image.open(io.BytesIO(image_bytes))
    
    if model is None:
        if model_path:
            model, class_names = load_image_model(model_path)
        else:
            return {'label': None, 'confidence': 0.0, 'note': 'No model available'}

    # Special handling for YOLOPredictor
    if hasattr(model, '__class__') and model.__class__.__name__ == 'YOLOPredictor':
        try:
            # YOLO can take PIL images directly
            results = model.model(img)
            res = results[0]
            
            # If it's classification, extract label and confidence from probs
            if hasattr(res, 'probs') and res.probs is not None:
                idx = int(res.probs.top1)
                conf = float(res.probs.top1conf)
                label = res.names[idx]
                return {'label': label, 'confidence': conf, 'sentence': label.capitalize() + '.'}
            
            # If it's detection, extract top detection or concatenated labels
            if hasattr(res, 'boxes') and len(res.boxes) > 0:
                # Use the detection with highest confidence
                box = res.boxes[0]
                idx = int(box.cls[0])
                conf = float(box.conf[0])
                label = res.names[idx]
                return {'label': label, 'confidence': conf, 'sentence': label.capitalize() + '.'}
                
            return {'label': None, 'confidence': 0.0, 'note': 'No detection in YOLO image'}
        except Exception as e:
            return {'label': None, 'confidence': 0.0, 'note': f'YOLO prediction error: {e}'}

    x = _preprocess_pil_image(img)
    if class_names is None:
        class_names = getattr(model, 'class_names', None) or _default_class_names()

    # Check model input shape and adjust if needed
    input_shape = model.input_shape if hasattr(model, 'input_shape') else (None, 224, 224, 3)
    if input_shape[1:3] == (28, 28):
        # Handle 28x28x1 grayscale (Sign MNIST)
        img_gray = img.convert('L')
        img_gray = img_gray.resize((28, 28))
        x = np.asarray(img_gray).astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=(0, -1))

    try:
        preds = model.predict(x)
        idx = int(np.argmax(preds, axis=1)[0])
        conf = float(np.max(preds))
        label = class_names[idx] if idx < len(class_names) else 'unknown'
        return {'label': label, 'confidence': conf, 'sentence': label.capitalize() + '.'}
    except Exception as e:
        return {'label': None, 'confidence': 0.0, 'note': f'Prediction error: {e}'}


def predict_from_bgr_frame(frame_bgr: np.ndarray, model_path: Optional[str] = None, model=None, class_names=None) -> dict:
    if model is None:
        if model_path:
            model, class_names = load_image_model(model_path)
        else:
            return {'label': None, 'confidence': 0.0, 'note': 'No model provided or model_path'}

    # Special handling for YOLOPredictor
    if hasattr(model, '__class__') and model.__class__.__name__ == 'YOLOPredictor':
        try:
            # For YOLO, we can pass the BGR frame directly
            results = model.model(frame_bgr)
            res = results[0]
            
            # If it's classification, extract label and confidence from probs
            if hasattr(res, 'probs') and res.probs is not None:
                idx = int(res.probs.top1)
                conf = float(res.probs.top1conf)
                label = res.names[idx]
                return {'label': label, 'confidence': conf, 'sentence': label.capitalize() + '.'}
            
            # If it's detection, extract top detection or concatenated labels
            if hasattr(res, 'boxes') and len(res.boxes) > 0:
                # Use the detection with highest confidence
                box = res.boxes[0]
                idx = int(box.cls[0])
                conf = float(box.conf[0])
                label = res.names[idx]
                return {'label': label, 'confidence': conf, 'sentence': label.capitalize() + '.', 'annotated_frame': res.plot()}
                
            return {'label': None, 'confidence': 0.0, 'note': 'No detection in YOLO frame'}
        except Exception as e:
            return {'label': None, 'confidence': 0.0, 'note': f'YOLO prediction error: {e}'}

    x = _preprocess_bgr_frame(frame_bgr)
    if class_names is None:
        class_names = getattr(model, 'class_names', None) or _default_class_names()
    try:
        preds = model.predict(x)
        idx = int(np.argmax(preds, axis=1)[0])
        conf = float(np.max(preds))
        label = class_names[idx] if idx < len(class_names) else 'unknown'
        return {'label': label, 'confidence': conf, 'sentence': label.capitalize() + '.'}
    except Exception as e:
        return {'label': None, 'confidence': 0.0, 'note': f'Prediction error: {e}'}
