import requests
import numpy as np
import pandas as pd
import cv2
import io
import os

def test_predict_csv_sample():
    # Path to the test dataset
    test_csv = 'd:/DL_Project/test.csv'
    
    if not os.path.exists(test_csv):
        print(f"Test dataset not found at {test_csv}. Please ensure it is in the correct location.")
        return

    print(f"Loading sample digit from {test_csv}...")
    try:
        # Load the first row of test.csv
        df = pd.read_csv(test_csv, nrows=1)
        pixel_data = df.values.reshape(28, 28).astype(np.uint8)
        
        # Resize to 64x64 or 200x200 (backend handles resizing)
        img_input = cv2.resize(pixel_data, (200, 200), interpolation=cv2.INTER_CUBIC)
        
        # Encode as PNG to send as file
        success, buffer = cv2.imencode(".png", img_input)
        if not success:
            print("Failed to encode image")
            return
            
        io_buf = io.BytesIO(buffer)
        
        # Send to backend
        url = "http://localhost:8000/predict"
        files = {"file": ("test_digit.png", io_buf, "image/png")}
        
        print(f"Sending test digit to {url}...")
        response = requests.post(url, files=files)
        
        print("Response Status:", response.status_code)
        if response.status_code == 200:
            result = response.json()
            print("\n--- Prediction Result ---")
            print(f"Predicted Digit: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Top Predictions:")
            for p in result.get('top_3', []):
                print(f"  Digit {p['prediction']}: {p['confidence']:.4f}")
        else:
            print("Error response:", response.text)
            print("Note: Make sure the backend server (main.py) is running.")
            
    except Exception as e:
        print("Test failed:", e)

if __name__ == "__main__":
    test_predict_csv_sample()
