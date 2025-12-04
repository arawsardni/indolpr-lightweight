import json
import os
import shutil

BASE_DIR = r"d:\File Gaung\Kuliah TIF UB\Semester 5\Deep Learning\Projek Akhir"
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def update_yolo_notebook():
    notebook_path = os.path.join(NOTEBOOKS_DIR, "Baseline_YOLO.ipynb")
    if not os.path.exists(notebook_path):
        print(f"Notebook not found: {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # Update model path
                if 'model = YOLO("yolo11n.pt")' in line:
                    line = line.replace('model = YOLO("yolo11n.pt")', 'model = YOLO("../models/yolo11n.pt")')
                # Update project path
                if 'project="yolo11_lpr"' in line:
                    line = line.replace('project="yolo11_lpr"', 'project="../results/yolo11_lpr"')
                new_source.append(line)
            cell['source'] = new_source
            
            # Clear outputs to reduce size and remove old paths in output
            cell['outputs'] = []
            cell['execution_count'] = None

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {notebook_path}")

def update_ocr_notebook():
    notebook_path = os.path.join(NOTEBOOKS_DIR, "Baseline_OCR.ipynb")
    if not os.path.exists(notebook_path):
        print(f"Notebook not found: {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # Update save_model_dir
                if 'save_model_dir: ./output/rec/lpr_finetune' in line:
                    line = line.replace('save_model_dir: ./output/rec/lpr_finetune', 'save_model_dir: ../results/ocr/lpr_finetune')
                
                # Update pretrained model path if we move it
                # Assuming we move en_PP-OCRv3_rec_train to models
                if 'en_PP-OCRv3_rec_train/best_accuracy' in line:
                     line = line.replace('en_PP-OCRv3_rec_train/best_accuracy', '../models/en_PP-OCRv3_rec_train/best_accuracy')
                
                new_source.append(line)
            cell['source'] = new_source
            
            # Clear outputs
            cell['outputs'] = []
            cell['execution_count'] = None

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {notebook_path}")

def move_artifacts():
    # Move PaddleOCR repo if exists
    src_paddle = os.path.join(NOTEBOOKS_DIR, "PaddleOCR")
    dst_paddle = os.path.join(BASE_DIR, "PaddleOCR")
    if os.path.exists(src_paddle):
        if os.path.exists(dst_paddle):
            print(f"Destination {dst_paddle} exists, skipping move.")
        else:
            shutil.move(src_paddle, dst_paddle)
            print(f"Moved {src_paddle} to {dst_paddle}")
            
    # Move OCR output if exists
    src_output = os.path.join(NOTEBOOKS_DIR, "output")
    dst_output = os.path.join(RESULTS_DIR, "ocr_output")
    if os.path.exists(src_output):
        if not os.path.exists(dst_output):
             shutil.move(src_output, dst_output)
             print(f"Moved {src_output} to {dst_output}")
        else:
             print(f"Destination {dst_output} exists, merging not implemented.")

    # Move pretrained weights if exists
    src_weights = os.path.join(NOTEBOOKS_DIR, "en_PP-OCRv3_rec_train")
    dst_weights = os.path.join(MODELS_DIR, "en_PP-OCRv3_rec_train")
    if os.path.exists(src_weights):
        if not os.path.exists(dst_weights):
            shutil.move(src_weights, dst_weights)
            print(f"Moved {src_weights} to {dst_weights}")

    src_tar = os.path.join(NOTEBOOKS_DIR, "en_PP-OCRv3_rec_train.tar")
    dst_tar = os.path.join(MODELS_DIR, "en_PP-OCRv3_rec_train.tar")
    if os.path.exists(src_tar):
        if not os.path.exists(dst_tar):
            shutil.move(src_tar, dst_tar)
            print(f"Moved {src_tar} to {dst_tar}")

if __name__ == "__main__":
    update_yolo_notebook()
    update_ocr_notebook()
    move_artifacts()
