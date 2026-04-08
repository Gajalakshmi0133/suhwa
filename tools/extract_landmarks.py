import os
import cv2
import mediapipe as mp
import numpy as np
import glob

def extract_landmarks():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    base_dirs = [
        'dataset/American_Sign_Language/train',
        'dataset/American_Sign_Language/asl_digits'
    ]
    
    out_dir = 'dataset/sequences'
    os.makedirs(out_dir, exist_ok=True)

    print("Extracting landmarks...")
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} not found, skipping.")
            continue
            
        classes = os.listdir(base_dir)
        for cls in classes:
            cls_path = os.path.join(base_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            
            print(f"Processing class: {cls}")
            images = glob.glob(os.path.join(cls_path, '*.jpg')) + glob.glob(os.path.join(cls_path, '*.png')) + glob.glob(os.path.join(cls_path, '*.jpeg'))
            
            # Limit to 200 images per class for faster training
            images = images[:200]
            
            for i, img_path in enumerate(images):
                out_name = f"{cls}_{i}.npy"
                out_path = os.path.join(out_dir, out_name)
                if os.path.exists(out_path):
                    continue

                if i % 50 == 0:
                    print(f"  {i}/{len(images)}")
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    lm_list = []
                    for lm in hand_landmarks.landmark:
                        lm_list.extend([lm.x, lm.y, lm.z])
                    
                    # Save as (1, 63) to match sequence format if needed, 
                    # but here we just save the flat array as we'll pad it later
                    # Actually, tools/prepare_dataset.py expects (T, D)
                    arr = np.array(lm_list).reshape(1, -1)
                    
                    out_name = f"{cls}_{i}.npy"
                    np.save(os.path.join(out_dir, out_name), arr)

    hands.close()
    print("Done!")

if __name__ == '__main__':
    extract_landmarks()
