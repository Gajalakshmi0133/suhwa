import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

def train_mnist_model(train_csv, test_csv, output_model_path='static/models/smnist.h5'):
    """
    Trains a CNN model on the Sign MNIST dataset from CSV files.
    """
    # Step 1: Load the dataset
    print(f"Loading data from {train_csv} and {test_csv}...")
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("Error: Dataset CSV files not found.")
        return

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    y_train = train_df['label']
    y_test = test_df['label']
    del train_df['label']
    del test_df['label']

    # Step 2: Preprocess labels (Label Binarization)
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    # Step 3: Reshape and normalize image data
    x_train = train_df.values
    x_test = test_df.values

    # Reshaping all of the MNIST training image files so the model understands the input files.
    # Normalizing pixel data (0-255) to (0-1).
    x_train = x_train / 255
    x_test = x_test / 255
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)

    # Step 4: Data Augmentation
    # Create the data generator to randomly implement changes to the data.
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )
    datagen.fit(x_train)

    # Step 5: Build CNN Model
    # Compiled to recognize 24 different classes (A-Z, skipping J and Z in standard Sign MNIST).
    model = Sequential()
    model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=24, activation='softmax'))

    # Step 6: Compile and Train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Starting training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=20,
        validation_data=(x_test, y_test)
    )

    # Step 7: Save the model
    print(f"Saving model to {output_model_path}...")
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.save(output_model_path)
    print("Training complete.")

if __name__ == '__main__':
    # Default paths for dataset
    train_path = 'dataset/sign_mnist_train.csv'
    test_path = 'dataset/sign_mnist_test.csv'
    
    train_mnist_model(train_path, test_path)
