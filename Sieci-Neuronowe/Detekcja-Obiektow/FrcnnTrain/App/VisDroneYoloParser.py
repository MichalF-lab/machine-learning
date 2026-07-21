import os
import torch
import numpy as np
from base_loader import GenericDetectionDataset

# YOLO VisDrone classes are 0-indexed
# FRCNN needs 1-based class ids (0 = background)
# YOLO 0..9  →  FRCNN 1..10
_YOLO_TO_FRCNN = {i: i + 1 for i in range(10)}


class VisDroneYoloDataset(GenericDetectionDataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        super().__init__(img_dir, transforms)
        self.label_dir = label_dir

    def __getitem__(self, idx):
        image = self.load_image(idx)
        img_h, img_w = image.shape[:2]

        stem       = os.path.splitext(self.img_names[idx])[0]
        label_path = os.path.join(self.label_dir, stem + '.txt')

        boxes, labels = [], []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    yolo_cls = int(parts[0])
                    if yolo_cls not in _YOLO_TO_FRCNN:
                        continue
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    xmin = max(0.0, (cx - bw / 2) * img_w)
                    ymin = max(0.0, (cy - bh / 2) * img_h)
                    xmax = min(float(img_w), (cx + bw / 2) * img_w)
                    ymax = min(float(img_h), (cy + bh / 2) * img_h)
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(_YOLO_TO_FRCNN[yolo_cls])

        boxes  = np.array(boxes,  dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.transforms:
            s      = self.transforms(image=image, bboxes=boxes, labels=labels)
            image  = s['image']
            boxes  = torch.tensor(s['bboxes'], dtype=torch.float32)
            labels = torch.tensor(s['labels'], dtype=torch.int64)
        else:
            image  = torch.from_numpy(image).permute(2, 0, 1)
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return image, {
            'boxes':    boxes,
            'labels':   labels,
            'image_id': torch.tensor([idx]),
            'area':     (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([0.0]),
            'iscrowd':  torch.zeros(len(labels), dtype=torch.int64),
        }
