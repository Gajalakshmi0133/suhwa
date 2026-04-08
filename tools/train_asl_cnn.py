import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import glob

def load_data(base_dir, img_size=(64, 64)):
    data = []
    labels = []
    
    # Define classes (A-Z)
    alphabet_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    # Define classes (0-9)
    digit_classes = [str(i) for i in range(10)]
    
    # Process alphabets
    for cls in alphabet_classes:
        cls_path = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_path):
            print(f"Warning: {cls_path} not found.")
            continue
        
        print(f"Loading alphabet class: {cls}")
        img_paths = glob.glob(os.path.join(cls_path, "*.jpg")) + \
                    glob.glob(os.path.join(cls_path, "*.jpeg")) + \
                    glob.glob(os.path.join(cls_path, "*.png"))
        
        # Limit images per class for efficiency
        img_paths = img_paths[:150] 
        
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, img_size)
            data.append(img)
            labels.append(cls)
            
    # Process digits
    digits_dir = os.path.join(base_dir, 'asl_digits')
    for cls in digit_classes:
        cls_path = os.path.join(digits_dir, cls)
        if not os.path.isdir(cls_path):
            print(f"Warning: {cls_path} not found.")
            continue
            
        print(f"Loading digit class: {cls}")
        img_paths = glob.glob(os.path.join(cls_path, "*.jpg")) + \
                    glob.glob(os.path.join(cls_path, "*.jpeg")) + \
                    glob.glob(os.path.join(cls_path, "*.png"))
        
        img_paths = img_paths[:150]
        
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, img_size)
            data.append(img)
            labels.append(cls)
            
    return np.array(data), np.array(labels)

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        layers.Input(shape=input_shape),
        # Normalization is part of the model now
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    base_dir = 'dataset/American_Sign_Language'
    img_size = (64, 64)
    
    print("Step 1: Loading data...")
    X, y = load_data(base_dir, img_size=img_size)
    print(f"Total samples: {len(X)}")
    
    if len(X) == 0:
        print("Error: No data loaded.")
        exit()
        
    print("Step 2: Preprocessing labels...")
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    num_classes = len(lb.classes_)
    print(f"Number of classes: {num_classes}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Data Augmentation layer to be used in training
    data_augmentation = Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
    
    print("Step 3: Building CNN model...")
    input_shape = (img_size[0], img_size[1], 3)
    model = Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Step 4: Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32
    )
    
    print("Step 5: Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Validation Accuracy: {acc*100:.2f}%")
    
    print("Step 6: Saving model...")
    os.makedirs('static/models', exist_ok=True)
    model.save('static/models/asl_cnn_model.h5')
    
    import pickle
    with open('static/models/asl_cnn_labels.pkl', 'wb') as f:
        pickle.dump(lb.classes_, f)
    
    print(f"Model saved to static/models/asl_cnn_model.h5")
    print(f"Classes: {lb.classes_}")
