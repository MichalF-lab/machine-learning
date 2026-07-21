import torch
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GenericDetectionDataset(Dataset):
    """
    Bazowa klasa dla wszystkich zbiorów danych.
    Implementuje wspólne ładowanie obrazów i aplikowanie transformacji.
    """
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg'))])

    def load_image(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # Normalizacja do [0, 1]
        return image

    def __getitem__(self, idx):
        # to jest tylko holder, będzie nadpisywane w poszczególnych parserach
        raise NotImplementedError("Należy zaimplementować metodę __getitem__ w klasie pochodnej")

    def __len__(self):
        return len(self.img_names)

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(640, 640),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def collate_fn(batch):
    return tuple(zip(*batch))