import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras.layers as layers
import keras.backend as K
import cv2


DATA_DIR = Path(__file__).resolve().parents[2] / 'dataset' / 'synthetic_sequences'
ALPHABET = list('abcdefghijklmnopqrstuvwxyz')
ALPHABET.append(' ')
# Map characters to 0..N-1, reserve last index (N) for the CTC blank
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1  # blank at index NUM_CLASSES-1


def load_sequences(data_dir=DATA_DIR, img_size=(64, 64)):
    seq_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    Xs = []
    Ys = []
    for d in seq_dirs:
        frames = sorted([p for p in d.iterdir() if p.name.startswith('frame_')])
        seq = []
        for f in frames:
            img = cv2.imread(str(f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, img_size)
            seq.append(img.astype(np.float32) / 255.0)
        Xs.append(np.stack(seq, axis=0))
        with open(d / 'transcript.txt', 'r', encoding='utf8') as tfp:
            txt = tfp.read().strip().lower()
        label = [CHAR_TO_IDX.get(ch, 0) for ch in txt]
        Ys.append(np.array(label, dtype=np.int32))
    return Xs, Ys


def pad_sequences_images(Xs, maxlen=None):
    if maxlen is None:
        maxlen = max(x.shape[0] for x in Xs)
    batch = []
    for x in Xs:
        pad = maxlen - x.shape[0]
        if pad > 0:
            pad_shape = (pad,) + x.shape[1:]
            pad_arr = np.zeros(pad_shape, dtype=x.dtype)
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


def build_model(img_size=(64, 64), time_steps=None):
    inp = layers.Input(shape=(time_steps, img_size[0], img_size[1], 1), name='input')
    x = layers.TimeDistributed(layers.Conv2D(16, 3, activation='relu'))(inp)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Conv2D(32, 3, activation='relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    y_pred = layers.Dense(NUM_CLASSES, activation='softmax', name='y_pred')(x)

    labels = layers.Input(name='labels', shape=(None,), dtype='int32')
    input_length = layers.Input(name='input_length', shape=(1,), dtype='int32')
    label_length = layers.Input(name='label_length', shape=(1,), dtype='int32')

    def ctc_lambda(args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    loss_out = layers.Lambda(ctc_lambda, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = tf.keras.models.Model(inputs=[inp, labels, input_length, label_length], outputs=loss_out)
    model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})
    # Inference model
    inf_model = tf.keras.models.Model(inputs=inp, outputs=y_pred)
    return model, inf_model


def train(epochs=5, batch_size=4):
    Xs, Ys = load_sequences()
    Xs = [x[..., None] for x in Xs]
    max_t = max(x.shape[0] for x in Xs)
    X = pad_sequences_images(Xs, maxlen=max_t)
    labels, label_lengths = pad_labels(Ys)
    input_lengths = np.ones((X.shape[0], 1), dtype=np.int32) * X.shape[1]

    model, inf_model = build_model(img_size=(X.shape[2], X.shape[3]), time_steps=X.shape[1])
    print(model.summary())

    # dummy y for loss lambda
    dummy_y = np.zeros((X.shape[0], 1))

    model.fit(x={'input': X, 'labels': labels, 'input_length': input_lengths, 'label_length': label_lengths},
              y=dummy_y,
              batch_size=batch_size,
              epochs=epochs)

    out_dir = Path(__file__).resolve().parents[2] / 'static' / 'models'
    out_dir.mkdir(parents=True, exist_ok=True)
    inf_model.save(out_dir / 'ctc_seq_model.h5')
    print('Saved inference model to', out_dir / 'ctc_seq_model.h5')


if __name__ == '__main__':
    train()
