"""
Simple Keras training script for temporal model (LSTM) using prepared arrays.
Usage:
  python tools/train_model.py --data dataset/prepared --epochs 20 --out models/sign_model.h5
"""
import os
import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model(T, D, num_classes):
    inp = keras.Input(shape=(T, D))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset/prepared')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--out', default='static/models/sign_model.h5')
    args = p.parse_args()
    X = np.load(os.path.join(args.data, 'X.npy'))
    y = np.load(os.path.join(args.data, 'y.npy'))
    labels = [l.strip() for l in open(os.path.join(args.data, 'labels.txt'), encoding='utf8').read().splitlines()]
    N, T, D = X.shape
    num_classes = len(labels)
    print('Data:', X.shape, 'classes:', num_classes)
    model = build_model(T, D, num_classes)
    callbacks = [keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor='val_loss')]
    model.fit(X, y, validation_split=0.1, epochs=args.epochs, batch_size=16, callbacks=callbacks)
    print('Saved model to', args.out)
