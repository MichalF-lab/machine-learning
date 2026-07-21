import os
import torch
import numpy as np
from .base_loader import GenericDetectionDataset

class VisDroneDataset(GenericDetectionDataset):
    def __init__(self, img_dir, annot_dir, transforms=None):
        super().__init__(img_dir, transforms)
        self.annot_dir = annot_dir
        
        self.class_map = {
            "1": 1,  # pedestrian
            "2": 2,  # people
            "3": 3,  # bicycle
            "4": 4,  # car
            "5": 5,  # van
            "6": 6,  # truck
            "7": 7,  # tricycle
            "8": 8,  # awning-tricycle
            "9": 9,  # bus
            "10": 10 # motor
        }

    def __getitem__(self, idx):
        image = self.load_image(idx)
        img_name = self.img_names[idx]
        
       
        annot_filename = img_name.replace(".jpg", ".txt")
        annot_path = os.path.join(self.annot_dir, annot_filename)
        
        boxes = []
        labels = []
        
        if os.path.exists(annot_path):
            with open(annot_path, 'r') as f:
                for line in f:
                    data = line.strip().split(',')
                    if len(data) < 6:
                        continue
                        
                    category = data[5]
                    
                    # wyrzucamy klasy other i ignored_regions
                    if category in self.class_map:
                        xmin = float(data[0])
                        ymin = float(data[1])
                        width = float(data[2])
                        height = float(data[3])
                        
                        xmax = xmin + width
                        ymax = ymin + height
                        
                        # zabezpieczenie przed boxami o zerowej powierzchni
                        if width > 0 and height > 0:
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