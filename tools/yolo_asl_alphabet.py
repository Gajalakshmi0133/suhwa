# Required pip install commands:
# pip install ultralytics opencv-python numpy

import cv2
import time
import numpy as np
from collections import deque
from ultralytics import YOLO

class ASLDetector:
    def __init__(self, model_path='runs/detect/train/weights/best.pt', conf_threshold=0.7, smoothing_frames=5):
        """
        Initialize the YOLOv11 ASL Alphabet Detector.
        """
        self.conf_threshold = conf_threshold
        self.smoothing_frames = smoothing_frames
        
        # Prediction smoothing queue
        self.prediction_history = deque(maxlen=smoothing_frames)
        
        # Load Model
        try:
            print(f"Loading YOLOv11 model from {model_path}...")
            self.model = YOLO(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_smoothed_prediction(self, current_prediction):
        """
        Returns a prediction only if it's consistent over multiple frames.
        """
        if current_prediction:
            self.prediction_history.append(current_prediction)
        else:
            self.prediction_history.clear()
            return None

        if len(self.prediction_history) == self.smoothing_frames:
            # Check if all items in history are the same
            if all(x == self.prediction_history[0] for x in self.prediction_history):
                return self.prediction_history[0]
        
        return None

    def run_realtime(self):
        """
        Main loop for real-time webcam detection.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        prev_time = 0
        
        print("Starting detection. Press 'q' to exit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                # 1. Run Inference
                results = self.model(frame, stream=True, verbose=False)
                
                top_prediction = None
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        conf = float(box.conf[0])
                        if conf >= self.conf_threshold:
                            # Extract data
                            cls = int(box.cls[0])
                            label = r.names[cls]
                            top_prediction = label
                            
                            # Draw Bounding Box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Prepare text
                            display_text = f"{label} {conf:.2f}"
                            cv2.putText(frame, display_text, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 2. Prediction Smoothing
                smoothed_label = self.get_smoothed_prediction(top_prediction)
                
                if smoothed_label:
                    status_text = f"Stable Prediction: {smoothed_label}"
                    cv2.putText(frame, status_text, (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                # 3. Calculate and Display FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 4. Show Frame
                cv2.imshow("SUHWA - YOLOv11 ASL Alphabet Detection", frame)

                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred during runtime: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Cleanup complete.")

if __name__ == "__main__":
    # Note: Using your specified path. 
    # If the file isn't there, you can pass a different path as an argument.
    MODEL_PATH = 'runs/detect/train/weights/best.pt'
    
    # Fallback to existing project model if best.pt is missing for demo purposes
    if not os.path.exists(MODEL_PATH):
        # Searching for alternative in common locations
        alternatives = ['static/models/best.onnx', 'best.onnx']
        for alt in alternatives:
            if os.path.exists(alt):
                MODEL_PATH = alt
                break

    import os
    detector = ASLDetector(model_path=MODEL_PATH)
    detector.run_realtime()
