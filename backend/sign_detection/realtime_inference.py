import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from backend.sign_detection.ctc_decode import decode_from_model, beam_search_with_lm
from backend.sign_detection.language_model import build_lm_from_file, build_word_lm_from_file


MODEL_PATH = Path(__file__).resolve().parents[2] / 'static' / 'models' / 'ctc_seq_model.h5'
IMG_SIZE = (64, 64)


def run_realtime(model_path=MODEL_PATH, cam_index=0, method='greedy', lm_path=None):
    if not model_path.exists():
        print('Model not found at', model_path)
        return

    print('Loading model from', model_path)
    model = tf.keras.models.load_model(str(model_path))
    # determine expected time length from model input if available
    try:
        time_steps_expected = int(model.input_shape[1])
    except Exception:
        time_steps_expected = 69

    buffer = deque(maxlen=time_steps_expected)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print('Cannot open camera', cam_index)
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    print('Starting webcam. Press q to quit.')
    last_pred = ''
    last_time = time.time()
    lm = None
    if lm_path:
        try:
            # prefer a word-level LM if available
            try:
                lm = build_word_lm_from_file(str(lm_path))
            except Exception:
                lm = build_lm_from_file(str(lm_path))
            print('Loaded LM from', lm_path)
        except Exception as e:
            print('Failed to load LM:', e)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # preprocess frame to match training preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, IMG_SIZE)
        norm = (small.astype(np.float32) / 255.0)[..., None]
        buffer.append(norm)

        pred_text = ''
        if len(buffer) == time_steps_expected:
            X = np.stack(list(buffer), axis=0)
            Xb = np.expand_dims(X, axis=0)
            try:
                if method == 'beam_lm' and lm is not None:
                    # get logits/probs from model
                    y_pred = model.predict(Xb)
                    pred_list = beam_search_with_lm(y_pred, lm, beam_width=10, top_paths=5, lm_weight=1.0)
                    pred_text = pred_list[0] if pred_list else ''
                else:
                    preds = decode_from_model(model, Xb, method='greedy' if method == 'greedy' else 'beam')
                    pred_text = preds[0] if preds else ''
            except Exception as e:
                pred_text = ''
        # smooth predictions: only update displayed text every 0.5s
        now = time.time()
        if pred_text and now - last_time > 0.5 and pred_text != last_pred:
            last_pred = pred_text
            last_time = now

        display_text = last_pred if last_pred else '<listening>'
        # overlay
        cv2.putText(frame, f'Prediction: {display_text}', (10, 30), font, 0.9, (0, 0, 255), 2)
        cv2.imshow('ASL Realtime', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_realtime()
