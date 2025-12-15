import cv2
import torch
import numpy as np
import argparse
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from src.ocr.lprnet import build_lprnet
from src.ocr.decoder import LPRLabelEncoder, CHARS

def load_lprnet(weights_path, device):
    lprnet = build_lprnet(class_num=len(CHARS)+1, dropout_rate=0)
    lprnet.to(device)
    
    if os.path.exists(weights_path):
        print(f"Loading LPRNet weights from {weights_path}")
        lprnet.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Warning: LPRNet weights not found at {weights_path}. Using random initialization.")
    
    lprnet.eval()
    return lprnet

def preprocess_lpr_input(img):
    # Resize to LPRNet input size (94, 24)
    img = cv2.resize(img, (94, 24))
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1)) # C, H, W
    return img

def run_inference(video_path, yolo_path, lpr_path, output_path=None, show=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Models
    print("Loading YOLO...")
    yolo = YOLO(yolo_path)
    
    print("Loading LPRNet...")
    lprnet = load_lprnet(lpr_path, device)
    decoder = LPRLabelEncoder(CHARS)

    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Video Writer
    out = None
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Starting inference loop...")
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # A. YOLO Detection
        results = yolo(frame, verbose=False)
        boxes = results[0].boxes
        
        lpr_batch_imgs = []
        lpr_batch_coords = [] # (x1, y1) to draw text later

        # B. Prepare crops for LPRNet
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Clip to frame
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Crop
                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size == 0: continue
                
                lpr_batch_imgs.append(preprocess_lpr_input(plate_img))
                lpr_batch_coords.append((x1, y1, x2, y2))

        # C. Run LPRNet Batch
        if lpr_batch_imgs:
            lpr_tensor = torch.from_numpy(np.array(lpr_batch_imgs)).to(device)
            
            with torch.no_grad():
                logits = lprnet(lpr_tensor) # (N, Class, Time)
                # preds = logits.cpu().numpy() # Decoder expects tensor or handles conversion logic specifically
            
            decoded_texts = decoder.decode_greedy(logits)
            
            # D. Draw Results
            for i, (text) in enumerate(decoded_texts):
                x1, y1, x2, y2 = lpr_batch_coords[i]
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Label Background
                (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_text, y1), (0, 255, 0), -1)
                
                # Draw Text
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Write/Show
        if out:
            out.write(frame)
        
        if show:
            cv2.namedWindow('LPR Pipeline', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('LPR Pipeline', 800, 600)
            cv2.imshow('LPR Pipeline', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    end_time = time.time()
    print(f"Processed {frame_count} frames in {end_time - start_time:.2f}s ({frame_count/(end_time - start_time):.2f} FPS)")

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Defaults
    default_video = os.path.join(project_root, "test_video.mp4")
    default_yolo = os.path.join(project_root, "artifacts", "yolo", "best.pt")
    default_lpr = os.path.join(project_root, "artifacts", "lprnet", "lprnet_best.pth")
    default_output = os.path.join(project_root, "output.mp4")

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=default_video, help="Path to input video")
    parser.add_argument("--yolo", type=str, default=default_yolo, help="Path to YOLO weights")
    parser.add_argument("--lpr", type=str, default=default_lpr, help="Path to LPRNet weights")
    parser.add_argument("--output", type=str, default=default_output, help="Path to output video")
    parser.add_argument("--no-show", action="store_true", help="Do not display window")
    
    args = parser.parse_args()
    
    run_inference(args.video, args.yolo, args.lpr, args.output, not args.no_show)
