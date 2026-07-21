import os
import json
import torch
import numpy as np
from .base_loader import GenericDetectionDataset

class BDD100KDataset(GenericDetectionDataset):
    def __init__(self, img_dir, annot_path, transforms=None):
        super().__init__(img_dir, transforms)

        with open(annot_path, 'r') as f:
            full_annotations = json.load(f)
            
        self.img_to_annot = {}
        valid_img_names = set(self.img_names) # set dla szybszego szukania
        
        for anno in full_annotations:
            if anno['name'] in valid_img_names:
                self.img_to_annot[anno['name']] = anno
                
        self.img_names = list(self.img_to_annot.keys())

        # Mapowanie klas BDD100K na ID
        self.class_map = {
            "pedestrian": 1,
            "rider": 2,
            "car": 3,
            "truck": 4,
            "bus": 5,
            "train": 6,
            "motorcycle": 7,
            "bicycle": 8,
            "traffic light": 9,
            "traffic sign": 10
        }

    def __getitem__(self, idx):
        image = self.load_image(idx)
        img_name = self.img_names[idx]
        anno_data = self.img_to_annot[img_name]
        
        boxes = []
        labels = []
        
        if 'labels' in anno_data:
            for obj in anno_data['labels']:
                # wybieramy tylko obiekty z bounding boxami (nie lane markings)
                if 'box2d' in obj:
                    category = obj['category']
                    if category in self.class_map:
                        box = obj['box2d']
                        xmin = float(box['x1'])
                        ymin = float(box['y1'])
                        xmax = float(box['x2'])
                        ymax = float(box['y2'])
                        
                        # zabezpieczenie przed boxami o zerowej powierzchni
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(self.class_map[category])

        
        boxes = np.array(boxes, dtype=np.float32) if len(boxes) > 0 else np.empty((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        
        if self.transforms:
            sample = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = sample['image']
            boxes = torch.tensor(sample['bboxes'], dtype=torch.float32)
            labels = torch.tensor(sample['labels'], dtype=torch.int64)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }

        return image, target