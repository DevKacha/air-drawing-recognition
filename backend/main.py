import os
import io
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI(title="Air Drawing AI API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and classes
MODEL_PATH = "model/sketch_model.h5"
CLASSES_PATH = "classes.npy"

model = None
classes = []

@app.on_event("startup")
def load_assets():
    global model, classes
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        try:
            model = load_model(MODEL_PATH)
            classes = np.load(CLASSES_PATH).tolist()
            print("Model and classes loaded successfully.")
        except Exception as e:
            print(f"Error loading model/classes: {e}")
    else:
        print("Model or classes file not found. Please train the model first.")

@app.get("/")
def read_root():
    return {"status": "Air Drawing AI API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, classes
    if model is None or not classes:
        # Try reloading if missing (e.g. model was trained after server started)
        if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
            model = load_model(MODEL_PATH)
            classes = np.load(CLASSES_PATH).tolist()
        else:
            return {"error": "Model not loaded"}
    
    # Read image contents
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return {"error": "Invalid image format"}

    # Find the bounding box of the drawing (non-zero pixels)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add a larger padding (around 20%) to match dataset distribution
        padding = int(max(w, h) * 0.20)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        # Crop
        img = img[y_start:y_end, x_start:x_end]
        
        # Slight dilation to fill gaps if the line is too thin
        kernel = np.ones((3,3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
    else:
        # If blank, just resize as is (all zeros)
        img = np.zeros((64, 64), dtype=np.uint8)

    # Resize to 64x64
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    
    # Model predict
    img_input = np.expand_dims(img, axis=-1)
    img_input = np.expand_dims(img_input, axis=0) # (1, 64, 64, 1)
    
    preds = model.predict(img_input, verbose=0)[0]
    
    # Get top 3 indices
    top_3_indices = np.argsort(preds)[-3:][::-1]
    results = [{"prediction": str(classes[i]), "confidence": float(preds[i])} for i in top_3_indices]
    
    # Return result
    return {
        "prediction": results[0]["prediction"],
        "confidence": results[0]["confidence"],
        "predictions": [results[0]["prediction"]], # Compatibility with multiple object prediction request
        "top_3": results
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
