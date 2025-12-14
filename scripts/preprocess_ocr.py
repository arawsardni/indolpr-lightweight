import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Paths
BASE_DIR = r"d:\File Gaung\Kuliah TIF UB\Semester 5\Deep Learning\Projek Akhir"
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "IndonesianLiscenePlateDataset", "plate_text_dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "dataset")
LABEL_FILE = os.path.join(DATASET_DIR, "label.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "preprocessed", "plate_text_cropped")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "dataset")
OUTPUT_LABEL_FILE = os.path.join(OUTPUT_DIR, "label.csv")

MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "yolo", "best.pt")

def preprocess_dataset():
    # Load Model
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Create Output Dirs
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    
    # Load Labels
    df = pd.read_csv(LABEL_FILE)
    print(f"Loaded {len(df)} samples from {LABEL_FILE}")
    
    new_data = []
    
    print("Preprocessing images (cropping plates)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        label = row['label']
        
        img_path = os.path.join(IMAGES_DIR, filename)
        
        if not os.path.exists(img_path):
            continue
        
        # Correction for specific file
        if filename == "6710UB.jpg":
             new_filename = "H6710UB.jpg"
             print(f"Correcting file {filename} to {new_filename}")
             filename = new_filename
             # Also update label in 'row' if necessary, but we use 'label' variable
             # If label is also wrong in CSV, we might need to hardcode it, 
             # but user said "according to its label", implying label might be correct or contained in filename?
             # "sesuai dengan labelnya menjadi H6710UB" -> Label is H6710UB.
             # Let's verify label. If the CSV label is wrong, I should fix it too?
             # User: "ganti nama file ... sesuai dengan labelnya menjadi H6710UB"
             # So I effectively treat it as H6710UB.
            
        # Custom Logic for Distorted/CCTV images (starting with digit)
        if filename[0].isdigit():
            original_img = cv2.imread(img_path)
            h, w = original_img.shape[:2]
            # Crop top half
            cropped_img = original_img[0:h//2, 0:w]
        else:
            # Run Inference for standard images
            original_img = cv2.imread(img_path)
            results = model(img_path, verbose=False)
            
            cropped_img = None
            
            # Get highest confidence box
            best_conf = -1
            best_box = None
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
            
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box)
                # Clip to image bounds
                h, w = original_img.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Crop
                cropped_img = original_img[y1:y2, x1:x2]
                
                # Fallback if crop is empty
                if cropped_img.size == 0:
                    cropped_img = original_img
            else:
                # If no detection, use original (maybe it's already tight, or model missed it)
                cropped_img = original_img
            
        # Save Cropped Image
        save_path = os.path.join(OUTPUT_IMAGES_DIR, filename)
        cv2.imwrite(save_path, cropped_img)
        
        new_data.append({'filename': filename, 'label': label})
        
    # Save New Label CSV
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(OUTPUT_LABEL_FILE, index=False)
    print(f"Saved processed dataset to {OUTPUT_DIR}")
    print(f"Total samples: {len(new_df)}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
    else:
        preprocess_dataset()
