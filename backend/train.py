import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = 64
DATASET_DIRS = [
    'd:/DL_Project/digits_updated',
    'd:/DL_Project/digits_jpeg'
]

def load_dataset_from_folders(dirs):
    images = []
    labels = []
    class_names = [str(i) for i in range(10)]
    
    print(f"Loading images from {len(dirs)} directories...")
    
    for base_dir in dirs:
        if not os.path.exists(base_dir):
            print(f"Directory not found: {base_dir}")
            continue
            
        print(f"Processing {base_dir}...")
        for label_name in os.listdir(base_dir):
            class_path = os.path.join(base_dir, label_name)
            if not os.path.isdir(class_path):
                continue
                
            # Class index (assuming folders are named '0', '1', etc.)
            try:
                class_idx = int(label_name)
            except ValueError:
                print(f"Skipping non-numeric folder: {label_name}")
                continue
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                # Load in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                # Resize to target
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                images.append(img_resized)
                labels.append(class_idx)
                
    return np.array(images), np.array(labels), class_names

# Load Folder-based Data
X, y, class_names = load_dataset_from_folders(DATASET_DIRS)

if X is not None and len(X) > 0:
    print(f"Successfully loaded {len(X)} images from folders.")
    # Normalize
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    num_classes = len(class_names)
    y_cat = to_categorical(y, num_classes)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.1, random_state=42)
    
    # Model architecture (consistent with main.py)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training Model on Folder data (Quick Update)...")
    # Training for 5 epochs for speed and accuracy
    model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_val, y_val))
    
    # Save model and classes
    os.makedirs('model', exist_ok=True)
    np.save('classes.npy', np.array(class_names))
    model.save('model/sketch_model.h5')
    print(f"New model saved based on folder dataset!")
    
else:
    print("Could not load any data from the specified folders.")

