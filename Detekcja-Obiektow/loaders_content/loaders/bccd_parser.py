import os
import json
import torch
import numpy as np
from .base_loader import GenericDetectionDataset

class BCCDDataset(GenericDetectionDataset):
    def __init__(self, img_dir, annot_dir, transforms=None):
        super().__init__(img_dir, transforms)
        self.annot_dir = annot_dir

        self.class_map = {
            "RBC": 1,
            "WBC": 2,
            "Platelets": 3
        }

    def __getitem__(self, idx):
        image = self.load_image(idx)
        img_name = self.img_names[idx]
        
        # zdjęcia mają nazwy BloodImage_00XXX.jpeg a adnotacje BloodImage_00XXX.jpeg.json
        annot_path = os.path.join(self.annot_dir, img_name + ".json")
        
        boxes = []
        labels = []
        
        if os.path.exists(annot_path):
            with open(annot_path, 'r') as f:
                data = json.load(f)
            
            for obj in data.get("objects", []):
                label_name = obj.get("classTitle")
                if label_name in self.class_map:
                    # objects->points->exterior -> [[x1, y1], [x2, y2]]
                    points = obj.get("points", {}).get("exterior", [])
                    if len(points) >= 2:
                        # [x1, y1] to pierwszy punkt, [x2, y2] to drugi
                        xmin, ymin = points[0]
                        xmax, ymax = points[1]
                        
                        # na wypadek gdyby punkty były zamienione kolejnością
                        x_min, x_max = min(xmin, xmax), max(xmin, xmax)
                        y_min, y_max = min(ymin, ymax), max(ymin, ymax)
                        
                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(self.class_map[label_name])


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