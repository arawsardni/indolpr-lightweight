import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from src.ocr.decoder import LPRLabelEncoder

class LPRDataset(Dataset):
    def __init__(self, img_dir, label_file, img_size=(94, 24), transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(label_file)
        self.img_size = img_size
        self.transform = transform
        self.encoder = LPRLabelEncoder()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        file_name, text = row['filename'], row['label']
        
        img_path = os.path.join(self.img_dir, file_name)
        image = cv2.imread(img_path)
        
        if image is None:
             # Placeholder for corrupted image, return a dummy
             image = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        # Resize
        image = cv2.resize(image, self.img_size)
        
        # Transform (HWC to CHW and Normalize)
        image = image.astype('float32')
        image -= 127.5
        image *= 0.0078125
        image = np.transpose(image, (2, 0, 1))

        if self.transform:
            image = self.transform(image)
            
        encoded_text = self.encoder.encode(text)
        
        return torch.from_numpy(image), torch.tensor(encoded_text), len(encoded_text)

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    
    for _, (img, label, length) in enumerate(batch):
        imgs.append(img)
        labels.extend(label)
        lengths.append(length)
        
    imgs = torch.stack(imgs, 0)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return imgs, labels, lengths
