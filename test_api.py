import requests
import numpy as np
import os
import json

def test_prediction():
    # Load a landmark sequence (A_0.npy)
    seq_file = 'dataset/sequences/A_0.npy'
    if not os.path.exists(seq_file):
        print(f"File {seq_file} not found")
        return

    arr = np.load(seq_file)
    # arr shape is (1, 63)
    
    # Convert to list for JSON
    landmarks = []
    hand = []
    for i in range(21):
        hand.append({
            'x': float(arr[0, i*3]),
            'y': float(arr[0, i*3+1]),
            'z': float(arr[0, i*3+2])
        })
    landmarks.append(hand)

    url = 'http://127.0.0.1:5000/api/detect-stream'
    payload = {
        'landmarks': landmarks,
        'lang': 'asl'
    }

    print("Sending request to API...")
    try:
        # Send multiple requests to trigger majority logic
        for _ in range(5):
            resp = requests.post(url, json=payload)
            print(f"Response: {resp.status_code}")
            print(f"Data: {resp.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    test_prediction()
