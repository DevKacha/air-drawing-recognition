# AI Air Drawing Full-Stack App (In-Air Hand-Drawn Numbers & Shapes OCR)

## Prerequisites

- Node.js (v18+)
- Python (3.12+)
- The "In-Air Hand-Drawn Number and Shape Dataset" containing `X.xlsx`, `Y.xlsx`, `label.xlsx` in their respective `/Numbers/Excel` and `/Shapes/Excel` inside `d:/DL_Project/`!

## Backend Setup

1. Open a terminal and navigate to the `backend` folder:
```bash
cd backend
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the CNN Model:
```bash
python train.py
```
This will take a few minutes. It will load images from the dataset, build a CNN, and save the model to `model/sketch_model.h5` and labels to `classes.npy`.

5. Run the API Server:
```bash
python main.py
```
The server will start at `http://localhost:8000`.

## Frontend Setup

1. Open a separate terminal and navigate to the `frontend` folder:
```bash
cd frontend
```

2. Install dependencies (if not fully completed or needed on another machine):
```bash
npm install
npm install @mediapipe/hands @mediapipe/camera_utils @mediapipe/drawing_utils lucide-react canvas-confetti
```

3. Start the Next.js dev server:
```bash
npm run dev
```

4. Open your browser and go to `http://localhost:3000`. 
- Allow camera access.
- Hold up your hand. MediaPipe will track your index finger tip.
- Draw an object in the air. Wait for 1.5 seconds without moving your hand significantly to let the system segment the drawing and send it to the backend.
- The top-3 predictions will be read out loud using text-to-speech and shown on the right-hand panel!

## Advanced Features Implemented

- real-time canvas overlay for drawing trajectory
- multi-object detection: the canvas clears after each prediction (simulate multiple objects)
- dark theme
- top-3 predictions rendering
- dynamic confidence bars
- voice output 
"# air-drawing-recognition" 
