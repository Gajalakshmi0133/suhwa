#!/usr/bin/env python3
"""
Train ASL image model on full images.

Uses ImageDataGenerator for data loading, MobileNetV2 as base model.
Trains on dataset/train/, validates on dataset/test/.
Saves model to static/models/asl_image_model.h5 with labels.txt.
"""
import os
import argparse
from pathlib import Path

import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

def build_model(num_classes, input_shape=(224, 224, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_class_names(train_dir):
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return classes

def main(args):
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names(str(train_dir))
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Num classes: {num_classes}")

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    model = build_model(num_classes)

    # Callbacks
    checkpoint = ModelCheckpoint(str(out_dir / 'asl_image_model.h5'), save_best_only=True, monitor='val_accuracy')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # Train
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // args.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // args.batch_size,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stop]
    )

    # Save labels
    with open(out_dir / 'labels.txt', 'w') as f:
        f.write('\n'.join(class_names))

    print(f"Model saved to {out_dir / 'asl_image_model.h5'}")
    print(f"Labels saved to {out_dir / 'labels.txt'}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train-dir', default='dataset/train')
    p.add_argument('--test-dir', default='dataset/test')
    p.add_argument('--out-dir', default='static/models')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    main(args)
