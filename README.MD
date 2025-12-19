# Indonesian License Plate Recognition (IndoLPR) - Lightweight

This repository contains an end-to-end pipeline for Indonesian license plate recognition, optimized for lightweight performance. It combines object detection (YOLOv11) and Optical Character Recognition (LPRNet + Beam Search) to detect and read license plates from video streams.

![Demo License Plate Detection](demo.gif)

## Features

*   **Two-Stage Pipeline**:
    1.  **Plate Detection**: Uses a fine-tuned **YOLOv11** model to locate license plates in frames.
    2.  **Text Recognition**: Uses **LPRNet** (License Plate Recognition Network) with **Beam Search Decoding** to accurately read characters from cropped plates.
*   **Beam Search Decoding**: Enhances OCR accuracy by exploring multiple possible character sequences instead of just the greedy best path.
*   **Video Inference**: Process full video files and generate annotated output videos.

## Project Structure

```
├── artifacts/          # Saved model weights (YOLO, LPRNet)
├── datasets/           # Raw and preprocessed datasets
├── pretrained-models/  # Pretrained models to be fine tuned
├── notebooks/          # Exploratory Data Analysis and Training Notebooks
├── results/            # Currently contains YOLO training result
├── scripts/            # Utility scripts for inference, preprocessing, etc.
├── src/                # Source code for core modules
│   ├── ocr/            # LPRNet architecture and decoding logic
│   └── evaluation/     # Metrics and evaluation tools
└── requirements.txt    # Python dependencies
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/arawsardni/indolpr-lightweight.git
    cd indolpr-lightweight
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Video Inference

To run the full recognition pipeline on a video file:

```bash
python scripts/run_video_inference.py --video test_video.mp4 --output output.mp4
```

**Arguments:**
*   `--video`: Path to the input video file.
*   `--yolo`: Path to the trained YOLO weights (default: `artifacts/yolo/best.pt`).
*   `--lpr`: Path to the trained LPRNet weights (default: `artifacts/lprnet/lprnet_best.pth`).
*   `--output`: Path to save the annotated output video.
*   `--show`: (Optional) Display the video window during processing.

### Demo Output

An example of the system's output can be found in `output.mp4`. This video demonstrates the pipeline's ability to detect plates and recognize text in real-world scenarios.

## Model Details

*   **Detection**: YOLOv11 (Nano) fine-tuned on the Indonesian License Plate Dataset.
*   **OCR**: LPRNet (lightweight CNN) trained with robust data augmentation and Beam Search decoding for high character accuracy (~86% Val Char Acc).

## Training

The training workflow is documented in the notebooks:
*   `notebooks/YOLO.ipynb`: Training the detection model.
*   `notebooks/OCR.ipynb`: Training the LPRNet recognition model.
