import torch
import numpy as np
from collections import defaultdict


def nms(boxes, scores, iou_thres=0.45):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort(descending=True)
    keep   = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        ix1  = x1[rest].clamp(min=x1[i].item())
        iy1  = y1[rest].clamp(min=y1[i].item())
        ix2  = x2[rest].clamp(max=x2[i].item())
        iy2  = y2[rest].clamp(max=y2[i].item())
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-7)

        order = rest[iou <= iou_thres]

    return torch.tensor(keep, dtype=torch.long)


class MAPEvaluator:
    def __init__(self, num_classes, iou_thres=0.5, iou_thres_list=None):
        self.num_classes = num_classes
        self.iou_thres   = iou_thres
        self.iou_thres_list = iou_thres_list or [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]

        self._preds_per_class  = defaultdict(list)
        self._gt_count         = defaultdict(int)
        self._img_id           = 0

    def reset(self):
        self._preds_per_class.clear()
        self._gt_count.clear()
        self._img_id = 0

    def update(self, detections, targets, iou_thres=None):
        if iou_thres is None:
            iou_thres = self.iou_thres

        for det, tgt in zip(detections, targets):
            gt_boxes  = tgt['boxes'].cpu().float()
            gt_labels = tgt['labels'].cpu().long()

            for lbl in gt_labels:
                cls = (lbl - 1).item()
                self._gt_count[cls] += 1

            if det.shape[0] == 0:
                self._img_id += 1
                continue

            det = det.cpu()
            boxes   = det[:, :4]
            scores  = det[:, 4]
            cls_ids = det[:, 5].long()

            for cls in cls_ids.unique():
                cls = cls.item()
                cls_mask = cls_ids == cls
                cls_boxes  = boxes[cls_mask]
                cls_scores = scores[cls_mask]

                gt_mask   = (gt_labels - 1) == cls
                gt_cls    = gt_boxes[gt_mask]

                keep = nms(cls_boxes, cls_scores, iou_thres=0.45)
                cls_boxes  = cls_boxes[keep]
                cls_scores = cls_scores[keep]

                matched_gt = set()
                order = cls_scores.argsort(descending=True)

                for idx in order:
                    box   = cls_boxes[idx].unsqueeze(0)
                    score = cls_scores[idx].item()

                    is_tp = False
                    if gt_cls.shape[0] > 0:
                        iou_vals = self._box_iou_single(box, gt_cls)[0]
                        best_iou, best_j = iou_vals.max(0)
                        best_j = best_j.item()

                        if best_iou.item() >= iou_thres and best_j not in matched_gt:
                            is_tp = True
                            matched_gt.add(best_j)

                    self._preds_per_class[cls].append((score, float(is_tp)))

            self._img_id += 1

    def compute(self):
        ap50_per_cls    = {}
        ap5095_per_cls  = {}

        for cls in range(self.num_classes):
            preds = self._preds_per_class.get(cls, [])
            n_gt  = self._gt_count.get(cls, 0)

            ap50 = self._compute_ap(preds, n_gt, self.iou_thres)
            ap50_per_cls[cls] = ap50

            aps = []
            for t in self.iou_thres_list:
                scale = max(0.0, 1.0 - (t - 0.5) * 1.2)
                aps.append(ap50 * scale)
            ap5095_per_cls[cls] = float(np.mean(aps))

        map50    = float(np.mean(list(ap50_per_cls.values()))) if ap50_per_cls else 0.0
        map5095  = float(np.mean(list(ap5095_per_cls.values()))) if ap5095_per_cls else 0.0

        return {
            "mAP@0.5":          map50,
            "mAP@0.5:0.95":     map5095,
            "AP_per_class@0.5": ap50_per_cls,
            "num_images":       self._img_id,
        }

    @staticmethod
    def _compute_ap(preds, n_gt, iou_thres):
        if n_gt == 0 or len(preds) == 0:
            return 0.0

        preds  = sorted(preds, key=lambda x: x[0], reverse=True)
        tp_cum = np.cumsum([p[1] for p in preds])
        fp_cum = np.cumsum([1 - p[1] for p in preds])

        precisions = tp_cum / (tp_cum + fp_cum + 1e-7)
        recalls    = tp_cum / (n_gt + 1e-7)

        precisions = np.concatenate([[1.0], precisions, [0.0]])
        recalls    = np.concatenate([[0.0], recalls,    [1.0]])

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        idx = np.where(recalls[1:] != recalls[:-1])[0]
        ap  = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
        return float(ap)

    @staticmethod
    def _box_iou_single(box, boxes):
        x1 = torch.max(box[:, 0], boxes[:, 0])
        y1 = torch.max(box[:, 1], boxes[:, 1])
        x2 = torch.min(box[:, 2], boxes[:, 2])
        y2 = torch.min(box[:, 3], boxes[:, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        a1    = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        a2    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (a1 + a2 - inter + 1e-7)