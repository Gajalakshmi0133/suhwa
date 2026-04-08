import cv2
import os
import sys
from ultralytics import YOLO

def run_yolo_asl(model_path='static/models/best.onnx'):
    """
    Standalone YOLOv11 ASL Detection script.
    Usage: python tools/yolo_demo.py [path_to_model]
    """
    if not os.path.exists(model_path):
        # Check if we are in tools directory or root
        if os.path.exists(os.path.join('..', model_path)):
            model_path = os.path.join('..', model_path)
        else:
            print(f"Model file not found: {model_path}")
            return

    print(f"Loading YOLOv11 model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press 'ESC' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run inference
        results = model(frame)
        
        # Visualize results on the frame
        annotated_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow('Suhwa - YOLOv11 ASL Detection', annotated_frame)
        
        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Default path relative to project root
    default_model = 'static/models/best.onnx'
    
    if len(sys.argv) > 1:
        default_model = sys.argv[1]
        
    run_yolo_asl(default_model)
