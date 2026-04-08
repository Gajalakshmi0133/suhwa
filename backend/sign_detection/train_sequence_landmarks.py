import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras.layers as layers
import keras.backend as K
import cv2

try:
    import mediapipe as mp
except Exception:
    mp = None

DATA_DIR = Path(__file__).resolve().parents[2] / 'dataset' / 'synthetic_sequences'

# Character mapping consistent with train_sequence_model
ALPHABET = list('abcdefghijklmnopqrstuvwxyz') + [' ']
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1


def extract_landmarks_from_bgr(frame):
    """Return a fixed-length landmark vector for up to 2 hands (21 pts x 3 coords each).
    If mediapipe not available or no hand detected, returns zeros vector.
    Output shape: (126,) -> 2 hands * 21 landmarks * 3 coords
    """
    vec = np.zeros((2, 21, 3), dtype=np.float32)
    if mp is None:
        return vec.reshape(-1)
    try:
        with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            if not res or not getattr(res, 'multi_hand_landmarks', None):
                return vec.reshape(-1)
            for i, hand in enumerate(res.multi_hand_landmarks[:2]):
                for j, lm in enumerate(hand.landmark[:21]):
                    vec[i, j, 0] = lm.x
                    vec[i, j, 1] = lm.y
                    vec[i, j, 2] = lm.z
            return vec.reshape(-1)
    except Exception:
        return vec.reshape(-1)


def load_sequences_landmarks(data_dir=DATA_DIR, img_size=(128, 128)):
    seq_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    Xs = []
    Ys = []
    for d in seq_dirs:
        frames = sorted([p for p in d.iterdir() if p.name.startswith('frame_')])
        seq = []
        for f in frames:
            img = cv2.imread(str(f))
            if img is None:
                feat = np.zeros((126,), dtype=np.float32)
            else:
                feat = extract_landmarks_from_bgr(img)
            seq.append(feat)
        Xs.append(np.stack(seq, axis=0))
        with open(d / 'transcript.txt', 'r', encoding='utf8') as tfp:
            txt = tfp.read().strip().lower()
        label = [CHAR_TO_IDX.get(ch, 0) for ch in txt]
        Ys.append(np.array(label, dtype=np.int32))
    return Xs, Ys


def pad_sequences_features(Xs, maxlen=None):
    if maxlen is None:
        maxlen = max(x.shape[0] for x in Xs)
    batch = []
    for x in Xs:
        pad = maxlen - x.shape[0]
        if pad > 0:
            pad_arr = np.zeros((pad,) + x.shape[1:], dtype=x.dtype)
            xpad = np.concatenate([x, pad_arr], axis=0)
        else:
            xpad = x[:maxlen]
        batch.append(xpad)
    return np.stack(batch, axis=0)


def pad_labels(Ys):
    maxlen = max(len(y) for y in Ys)
    labels = np.zeros((len(Ys), maxlen), dtype=np.int32)
    label_lengths = np.zeros((len(Ys),), dtype=np.int32)
    for i, y in enumerate(Ys):
        labels[i, :len(y)] = y
        label_lengths[i] = len(y)
    return labels, label_lengths


def build_model(time_steps=None, feature_dim=126):
    inp = layers.Input(shape=(time_steps, feature_dim), name='input')
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    y_pred = layers.Dense(NUM_CLASSES, activation='softmax', name='y_pred')(x)

    labels = layers.Input(name='labels', shape=(None,), dtype='int32')
    input_length = layers.Input(name='input_length', shape=(1,), dtype='int32')
    label_length = layers.Input(name='label_length', shape=(1,), dtype='int32')

    def ctc_lambda(args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    loss_out = layers.Lambda(ctc_lambda, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = tf.keras.models.Model(inputs=[inp, labels, input_length, label_length], outputs=loss_out)
    model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})
    inf_model = tf.keras.models.Model(inputs=inp, outputs=y_pred)
    return model, inf_model


def train(epochs=10, batch_size=4):
    print('Loading landmark sequences...')
    Xs, Ys = load_sequences_landmarks()
    max_t = max(x.shape[0] for x in Xs)
    X = pad_sequences_features(Xs, maxlen=max_t)
    labels, label_lengths = pad_labels(Ys)
    input_lengths = np.ones((X.shape[0], 1), dtype=np.int32) * X.shape[1]

    model, inf_model = build_model(time_steps=X.shape[1], feature_dim=X.shape[2])
    print(model.summary())

    dummy_y = np.zeros((X.shape[0], 1))
    model.fit(x={'input': X, 'labels': labels, 'input_length': input_lengths, 'label_length': label_lengths},
              y=dummy_y,
              batch_size=batch_size,
              epochs=epochs)

    out_dir = Path(__file__).resolve().parents[2] / 'static' / 'models'
    out_dir.mkdir(parents=True, exist_ok=True)
    inf_model.save(out_dir / 'ctc_landmark_model.h5')
    print('Saved landmark inference model to', out_dir / 'ctc_landmark_model.h5')


if __name__ == '__main__':
    train()
