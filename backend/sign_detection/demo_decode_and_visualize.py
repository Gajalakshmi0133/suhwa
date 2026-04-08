import os
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import importlib.util

# Load ctc_decode module directly to avoid package import issues when running as script
mod_path = Path(__file__).resolve().parent / 'ctc_decode.py'
spec = importlib.util.spec_from_file_location('ctc_decode_local', str(mod_path))
ctc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctc_mod)
decode_from_model = ctc_mod.decode_from_model


SYN_DIR = Path(__file__).resolve().parents[2] / 'dataset' / 'synthetic_sequences'
MODEL_PATH = Path(__file__).resolve().parents[2] / 'static' / 'models' / 'ctc_seq_model.h5'


def load_sequence(seq_dir, img_size=(64, 64)):
    frames = sorted([p for p in seq_dir.iterdir() if p.name.startswith('frame_')])
    imgs = []
    for f in frames:
        im = cv2.imread(str(f))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, img_size)
        imgs.append(im.astype(np.float32) / 255.0)
    X = np.stack(imgs, axis=0)
    return X[..., None]


def visualize_prediction(seq_dir, pred_text, gt_text):
    first_frame = sorted([p for p in seq_dir.iterdir() if p.name.startswith('frame_')])[0]
    img = cv2.imread(str(first_frame))
    vis = img.copy()
    cv2.putText(vis, f'GT: {gt_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, f'PRED: {pred_text}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    out = seq_dir / 'prediction.jpg'
    cv2.imwrite(str(out), vis)


def run_demo(model_path=MODEL_PATH, method='beam'):
    if not model_path.exists():
        print('Model not found at', model_path)
        return
    model = tf.keras.models.load_model(str(model_path))
    # expected time steps from model input
    time_expected = None
    try:
        time_expected = int(model.input_shape[1])
    except Exception:
        time_expected = None
        lm_path=None):
        seq_dirs = sorted([d for d in SYN_DIR.iterdir() if d.is_dir()])
        lm = None
        if lm_path:
            try:
                from backend.sign_detection.language_model import build_word_lm_from_file
                lm = build_word_lm_from_file(str(lm_path))
            except Exception:
                try:
                    from backend.sign_detection.language_model import build_lm_from_file
                    lm = build_lm_from_file(str(lm_path))
                except Exception as e:
                    print('Failed to load LM:', e)
    for d in seq_dirs:
        X = load_sequence(d)
        # pad or truncate to model expected time dimension if available
        if time_expected is not None:
            t = X.shape[0]
            if t < time_expected:
                pad_count = time_expected - t
                pad_arr = np.zeros((pad_count, X.shape[1], X.shape[2], X.shape[3]), dtype=X.dtype)
                X = np.concatenate([X, pad_arr], axis=0)
            elif t > time_expected:
                X = X[:time_expected]
        print('Loaded sequence', d.name, 'X shape', X.shape)
        Xb = np.expand_dims(X, axis=0)
        print('Batched X shape', Xb.shape)
            if lm is not None and method in ('beam', 'beam_lm'):
                # run model to get probabilities and rescore
                y_pred = model.predict(Xb)
                preds_list = beam_search_with_lm(y_pred, lm, beam_width=10, top_paths=5, lm_weight=1.0)
                pred = preds_list[0] if preds_list else ''
            else:
                preds = decode_from_model(model, Xb, method=method)
                pred = preds[0]
        with open(d / 'transcript.txt', 'r', encoding='utf8') as f:
            gt = f.read().strip().lower()
        print(d.name, 'GT:', gt, 'PRED:', pred)
        visualize_prediction(d, pred, gt)


if __name__ == '__main__':
    run_demo()
