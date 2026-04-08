import argparse
import os
import string

import numpy as np
import pandas as pd
import tensorflow as tf


def load_data(path):
    df = pd.read_csv(path)
    # Map labels to 0..23 (remove 'j' and 'z' from alphabet if dataset used those indices)
    y = np.array([label if label < 9 else label - 1 for label in df['label']])
    df = df.drop('label', axis=1)
    x = np.array([df.iloc[i].to_numpy().reshape((28, 28)) for i in range(len(df))]).astype(np.float32)
    x = np.expand_dims(x, axis=3)
    # One-hot encoding
    y = pd.get_dummies(y).values
    return x, y


def build_model(input_shape=(28, 28, 1), num_classes=24):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main(args):
    print('Loading training data...')
    X_train, Y_train = load_data(args.train_csv)
    if args.test_csv and os.path.exists(args.test_csv):
        X_test, Y_test = load_data(args.test_csv)
    else:
        X_test, Y_test = None, None

    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    print('Shapes:', X_train.shape, Y_train.shape)

    model = build_model(input_shape=X_train.shape[1:], num_classes=Y_train.shape[1])
    model.summary()

    callbacks = []
    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor='val_loss')
    callbacks.append(ckpt)

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=args.epochs, batch_size=args.batch_size)

    # Save labels (class names)
    class_names = list(string.ascii_lowercase[:26])
    # remove 'j' and 'z' if dataset used same mapping as common sign datasets
    class_names.remove('j')
    class_names.remove('z')
    labels_path = os.path.join(out_dir, 'labels.txt')
    with open(labels_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(class_names))

    # Attach class_names to model for convenience then save
    try:
        model.class_names = class_names
    except Exception:
        pass

    # If ModelCheckpoint saved a best model, that's at args.out. Ensure final model exists.
    if not os.path.exists(args.out):
        model.save(args.out)

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_csv = os.path.join(out_dir, 'training_history.csv')
    hist_df.to_csv(hist_csv, index=False)
    print('Saved model to', args.out)
    print('Saved labels to', labels_path)
    print('Saved training history to', hist_csv)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train-csv', dest='train_csv', required=True, help='Path to sign_mnist_train.csv')
    p.add_argument('--test-csv', dest='test_csv', default=None, help='Optional sign_mnist_test.csv')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--out', default='static/models/sign_mnist_cnn.h5')
    args = p.parse_args()
    main(args)
