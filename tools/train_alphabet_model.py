import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

def extract_landmarks(base_dir):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    data = []
    labels = []
    
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} not found.")
        return np.array([]), np.array([])

    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    # Exclude 'test', 'train', 'asl_digits' if they are metadata folders
    classes = [c for c in classes if c not in ['test', 'train', 'asl_digits']]
    
    print(f"Classes found: {classes}")
    
    for cls in classes:
        cls_path = os.path.join(base_dir, cls)
        print(f"Processing class: {cls}")
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            import glob
            images.extend(glob.glob(os.path.join(cls_path, ext)))
        
        # Limit to 600 images per class for balanced and reasonably fast training
        images = images[:600]
        
        for i, img_path in enumerate(images):
            image = cv2.imread(img_path)
            if image is None: continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm_list = []
                    # Get wrist coordinates for normalization
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y
                    wrist_z = hand_landmarks.landmark[0].z
                    
                    # Also get max distance for scaling
                    max_dist = 0
                    temp_lms = []
                    for lm in hand_landmarks.landmark:
                        dx, dy, dz = lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                        if dist > max_dist: max_dist = dist
                        temp_lms.append([dx, dy, dz])
                    
                    if max_dist > 0:
                        for lm in temp_lms:
                            lm_list.extend([lm[0]/max_dist, lm[1]/max_dist, lm[2]/max_dist])
                        
                        data.append(lm_list)
                        labels.append(cls)
            
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i} images...")
                
    hands.close()
    return np.array(data), np.array(labels)

def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    base_dataset_dir = 'dataset/American_Sign_Language'
    
    print("Step 1: Extracting landmarks...")
    X, y = extract_landmarks(base_dataset_dir)
    print(f"Extracted {len(X)} samples.")
    
    if len(X) == 0:
        print("No landmarks extracted. Check dataset path.")
        exit()
        
    print("Step 2: Preprocessing...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # Save label classes for inference
    os.makedirs('static/models', exist_ok=True)
    with open('static/models/alphabet_labels.pkl', 'wb') as f:
        pickle.dump(label_encoder.classes_, f)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print("Step 3: Training model...")
    model = build_model(X.shape[1], num_classes)
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    print("Step 4: Saving model...")
    model.save('static/models/alphabet_landmark_model.h5')
    print("Training complete. Model saved to static/models/alphabet_landmark_model.h5")
    print(f"Classes: {label_encoder.classes_}")
