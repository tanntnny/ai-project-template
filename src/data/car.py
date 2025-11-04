from typing import Callable

from ..core.protocol import DataProtocol

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CarDataset(Dataset):
    def __init__(self, src, transform: transforms.Compose = None):
        self.data = pd.read_csv(src)
        self.samples = []
        self.transform = transform

        for _, row in self.data.iterrows():
            img_path = row["file"]
            label = row["label"]
            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f, label = self.samples[idx]

        with Image.open(f) as img:
            image = img.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        "x": images,
        "labels": labels
    }

class CarDataProtocol(DataProtocol):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.train_dataset = CarDataset(self.cfg.data.train, self.transform)
        self.val_dataset = CarDataset(self.cfg.data.val, self.transform)
    
    def get_collator(self) -> Callable | None:
        return collate_fn
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_eval_dataset(self) -> Dataset:
        return self.val_dataset
    
    def get_test_dataset(self) -> Dataset | None:
        return None