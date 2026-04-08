import os
import sys
# Add current directory to path
sys.path.append(os.getcwd())

from backend.sign_detection.model_loader import load_model
from config import Config

model_path = os.path.join(os.getcwd(), 'static', 'models', 'asl_image_model.h5')
print(f"Checking model at: {model_path}")
print(f"Exists: {os.path.exists(model_path)}")

m = load_model(model_path)
if m:
    print("Model loaded successfully")
    labels_p = os.path.join(os.path.dirname(model_path), 'labels.txt')
    if os.path.exists(labels_p):
        with open(labels_p, 'r') as f:
            class_names = [l.strip() for l in f.readlines() if l.strip()]
            print(f"Loaded {len(class_names)} labels from labels.txt")
            m.class_names = class_names
    
    print(f"Class names: {getattr(m, 'class_names', 'None')}")
else:
    print("Model FAILED to load")
