import requests
import numpy as np
import pandas as pd
import cv2
import io
import os

def test_mnist_prediction():
    # Load a few samples from test.csv
    test_csv = 'd:/DL_Project/test.csv'
    if not os.path.exists(test_csv):
        print("test.csv not found")
        return
        
    print(f"Loading sample images from {test_csv}...")
    df = pd.read_csv(test_csv, nrows=5)
    pixel_data = df.values
    images_28 = pixel_data.reshape(-1, 28, 28).astype(np.uint8)
    
    url = "http://localhost:8000/predict"
    
    for i in range(len(images_28)):
        img = images_28[i]
        # Resize to 640x640 just to simulate air drawing canvas size if needed,
        # but the backend will resize it to 64x64 anyway.
        # Let's send it as is or slightly larger to see flexibility.
        img_large = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
        
        # Encode as PNG
        _, buffer = cv2.imencode(".png", img_large)
        io_buf = io.BytesIO(buffer)
        
        # Send to backend
        print(f"Sending Sample {i+1} to {url}...")
        files = {"file": (f"sample_{i}.png", io_buf, "image/png")}
        
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                print(f"Sample {i+1} Prediction: {response.json()['prediction']} (Confidence: {response.json()['confidence']:.2f})")
            else:
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_mnist_prediction()
