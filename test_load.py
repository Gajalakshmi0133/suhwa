import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from backend.sign_detection.model_loader import load_model
from config import Config

model_path = Config.MODEL_PATH
print(f"Loading model from: {model_path}")
if os.path.exists(model_path):
    print("File exists.")
else:
    print("File DOES NOT exist.")

model = load_model(model_path)
if model:
    print("Model loaded successfully!")
    try:
        print(f"Input shape: {model.input_shape}")
    except:
        pass
else:
    print("Model failed to load.")
