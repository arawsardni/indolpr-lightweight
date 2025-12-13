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

        self.images = []
        self.encoded_labels = []
        self.lengths = []

        print("Preloading images into RAM...")

        for _, row in self.labels.iterrows():
            img_path = os.path.join(self.img_dir, row['filename'])
            image = cv2.imread(img_path)

            if image is None:
                image = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

            image = cv2.resize(image, img_size)
            image = image.astype(np.float32)
            image -= 127.5
            image *= 0.0078125
            image = np.transpose(image, (2, 0, 1))

            self.images.append(torch.from_numpy(image))

            encoded = self.encoder.encode(row['label'])
            self.encoded_labels.append(torch.tensor(encoded, dtype=torch.long))
            self.lengths.append(len(encoded))

        print("Preload done.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.encoded_labels[idx], self.lengths[idx]


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
