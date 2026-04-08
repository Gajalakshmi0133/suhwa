import os


try:
    from ultralytics import YOLO
    import numpy as np

    class YOLOPredictor:
        def __init__(self, model):
            self.model = model
            self.class_names = list(model.names.values()) if hasattr(model, 'names') else []

        def __call__(self, *args, **kwargs):
            return self.model(*args, **kwargs)

        def predict(self, x_numpy):
            # x_numpy expected shape: (N, H, W, C)
            # YOLO expects a list of images or a single image
            results = self.model(x_numpy)
            all_probs = []
            for res in results:
                if hasattr(res, 'probs') and res.probs is not None:
                    all_probs.append(res.probs.data.cpu().numpy())
                else:
                    # If it's a detection model, we might want scores or results
                    pass
            if all_probs:
                return np.array(all_probs)
            return results
except ImportError:
    pass


def load_model(model_path):
    """
    Try to load a Keras model when a path is provided. Import TensorFlow lazily
    so the rest of the app can run in environments without TensorFlow installed.
    Returns the model or None on failure.
    """
    if not model_path or not os.path.exists(model_path):
        return None

    # Try YOLOv11 / Ultralytics models
    if model_path.endswith('.onnx') or model_path.endswith('.pt'):
        try:
            model = YOLO(model_path)
            return YOLOPredictor(model)
        except Exception as e_yolo:
            print('Failed to load YOLO model:', e_yolo)

    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow not available, skipping model load:", e)
        # Try PyTorch / HuggingFace ViT-style models as a fallback
        try:
            import torch
            from PIL import Image
            import numpy as np
            from transformers import ViTFeatureExtractor, ViTForImageClassification

            class TorchViTPredictor:
                def __init__(self, model, feature_extractor, device='cpu'):
                    self.model = model
                    self.fe = feature_extractor
                    self.device = torch.device(device)
                    self.model.to(self.device)
                    self.model.eval()
                    self.class_names = [] # type: ignore

                def predict(self, x_numpy):
                    # x_numpy expected shape: (N, H, W, C) with floats 0-1 or uint8
                    imgs = []
                    for arr in x_numpy:
                        a = arr
                        if a.dtype != np.uint8:
                            a = (a * 255.0).astype('uint8')
                        # squeeze channel dim if single-channel
                        if a.ndim == 3 and a.shape[2] == 1:
                            a = a[:, :, 0]
                        if a.ndim == 2:
                            pil = Image.fromarray(a).convert('RGB')
                        else:
                            pil = Image.fromarray(a).convert('RGB')
                        imgs.append(pil)

                    inputs = self.fe(images=imgs, return_tensors='pt')
                    pixel_values = inputs['pixel_values'].to(self.device)
                    with torch.no_grad():
                        outputs = self.model(pixel_values)
                        logits = outputs.logits.cpu().numpy()
                    # convert logits to probabilities similar to Keras' predict
                    try:
                        import scipy.special as _ss
                        probs = _ss.softmax(logits, axis=1)
                    except Exception:
                        # fallback softmax
                        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                        probs = exp / np.sum(exp, axis=1, keepdims=True)
                    return probs

            # Determine how to construct a ViT model from the provided path
            # If it's a directory, try loading a HF model directly
            if os.path.isdir(model_path):
                fe = ViTFeatureExtractor.from_pretrained(model_path)
                model = ViTForImageClassification.from_pretrained(model_path)
                return TorchViTPredictor(model, fe)

            # If model_path is a file, try to instantiate a pretrained ViT and load weights
            if os.path.isfile(model_path):
                # Try to load a state_dict
                try:
                    sd = torch.load(model_path, map_location='cpu')
                except Exception as e2:
                    print('Failed to torch.load checkpoint:', e2)
                    return None

                # Instantiate base ViT and attempt to load state dict
                try:
                    base_name = 'google/vit-base-patch16-224-in21k'
                    fe = ViTFeatureExtractor.from_pretrained(base_name)
                    model = ViTForImageClassification.from_pretrained(base_name, ignore_mismatched_sizes=True)
                    # If checkpoint is a Lightning checkpoint, extract state_dict
                    if isinstance(sd, dict) and 'state_dict' in sd:
                        sd = sd['state_dict']
                    # attempt to load state dict non-strictly
                    try:
                        model.load_state_dict(sd, strict=False)
                    except Exception:
                        # try loading keys with common prefixes stripped
                        new_sd = {}
                        for k, v in sd.items():
                            nk = k
                            if k.startswith('model.'):
                                nk = k[len('model.'):]
                            if k.startswith('vit.'):
                                nk = k[len('vit.'):]
                            new_sd[nk] = v
                        try:
                            model.load_state_dict(new_sd, strict=False)
                        except Exception:
                            pass
                    return TorchViTPredictor(model, fe)
                except Exception as e:
                    print('Failed to build ViT predictor from file:', e)
                    return None
        except Exception as e_torch:
            print('PyTorch/Transformers not available, skipping PyTorch model load:', e_torch)
            return None

    # If TensorFlow import succeeded, try loading a Keras model
    try:
        import keras
        custom_objects = {
            'NotEqual': tf.math.not_equal,
            'tf': tf,
        }
        
        try:
            model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            return model
        except TypeError as e:
            if "Missing required positional argument" in str(e):
                print(f"Model has compatibility issues. Attempting legacy loading approach...")
                try:
                    model = keras.models.load_model(model_path, compile=False)
                    return model
                except Exception as e2:
                    print(f"Legacy loading also failed: {e2}")
                    return None
            else:
                raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Failed to load model:", e)
        return None
