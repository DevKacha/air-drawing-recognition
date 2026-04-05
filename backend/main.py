import os
import io
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

tensorflow_available = True
try:
    from tensorflow.keras.models import load_model
except ImportError as e:
    tensorflow_available = False
    load_model = None
    print(f"TensorFlow import failed: {e}. Install tensorflow or tensorflow-cpu.")

app = FastAPI(title="Air Drawing AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/sketch_model.h5"
CLASSES_PATH = "classes.npy"

model = None
classes = []

@app.on_event("startup")
def load_assets():
    global model, classes
    if not tensorflow_available:
        print("TensorFlow not available; model loading is disabled. Install tensorflow-cpu or tensorflow.")
        return

    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        try:
            model = load_model(MODEL_PATH)
            classes = np.load(CLASSES_PATH).tolist()
            print("Model and classes loaded successfully.")
        except Exception as e:
            print(f"Error loading model/classes: {e}")
    else:
        print("Model or classes file not found. Please train the model first.")

@app.get("/check")
def check_root():
    return {
        "status": "Air Drawing AI API is running",
        "tensorflow_available": tensorflow_available,
        "model_loaded": model is not None and bool(classes),
        "classes_count": len(classes),
    }

@app.get("/")
def read_root():
    return {"status": "Air Drawing AI API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, classes

    if not tensorflow_available:
        raise HTTPException(status_code=503, detail="TensorFlow is not installed. Install tensorflow-cpu or tensorflow.")

    if model is None or not classes:
        if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
            model = load_model(MODEL_PATH)
            classes = np.load(CLASSES_PATH).tolist()
        else:
            raise HTTPException(status_code=503, detail="Model not loaded. Train model first with python train.py.")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    
    img_bgra = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_bgra is None:
        return {"error": "Invalid image format"}
    
    if img_bgra.ndim == 2:
        img = img_bgra
    elif img_bgra.shape[2] == 4:
        alpha = img_bgra[:, :, 3].astype(np.float32) / 255.0
        stroke = img_bgra[:, :, 0].astype(np.float32)
        img = (stroke * alpha).astype(np.uint8)
    else:
        img = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2GRAY)
    
    if img is None:
        return {"error": "Invalid image format"}

    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        padding = int(max(w, h) * 0.20)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        img = img[y_start:y_end, x_start:x_end]
        
        kernel = np.ones((3,3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
    else:
        img = np.zeros((64, 64), dtype=np.uint8)

    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    
    img_input = np.expand_dims(img, axis=-1)
    img_input = np.expand_dims(img_input, axis=0) # (1, 64, 64, 1)
    
    preds = model.predict(img_input, verbose=0)[0]
    
    top_3_indices = np.argsort(preds)[-3:][::-1]
    results = [{"prediction": str(classes[i]), "confidence": float(preds[i])} for i in top_3_indices]
    
    return {
        "prediction": results[0]["prediction"],
        "confidence": results[0]["confidence"],
        "predictions": [results[0]["prediction"]],
        "top_3": results
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)