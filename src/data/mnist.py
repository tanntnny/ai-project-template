from __future__ import annotations


import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from typing import Callable

from ..interfaces.protocol import DataProtocol

class MNISTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {
            "x": image,
            "labels": torch.tensor(label, dtype=torch.long)
        }

class MNISTData(DataProtocol):
    def __init__(self, cfg):
        self.cfg = cfg

        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.MNIST(self.cfg.data.root, train=True, download=True, transform=tf)
        test_dataset = datasets.MNIST(self.cfg.data.root, train=False, download=True, transform=tf)
        
        train_size = len(train_dataset)
        eval_size = int(train_size * self.cfg.data.train_ratio)
        train_size = train_size - eval_size
        
        indices = torch.randperm(len(train_dataset))
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]
        
        self.train_dataset = MNISTDataset(Subset(train_dataset, train_indices))
        self.eval_dataset = MNISTDataset(Subset(train_dataset, eval_indices))
        self.test_dataset = MNISTDataset(test_dataset)

    def get_collator(self) -> Callable | None:
        return None

    def get_train_dataset(self) -> Dataset:
        return self.train_dataset

    def get_eval_dataset(self) -> Dataset:
        return self.eval_dataset

    def get_test_dataset(self) -> Dataset:
        return self.test_dataset