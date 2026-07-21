import torch
import torch.nn as nn
import torch.nn.functional as F


def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    px1, py1, px2, py2 = pred_boxes.unbind(dim=-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(dim=-1)

    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    p_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    t_area = (tx2 - tx1).clamp(0) * (tx2 - tx1).clamp(0)
    union  = p_area + t_area - inter + eps
    iou    = inter / union

    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2)
    ey2 = torch.max(py2, ty2)
    c2  = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps

    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2
    d2  = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    pw  = (px2 - px1).clamp(0)
    ph  = (py2 - py1).clamp(0)
    tw  = (tx2 - tx1).clamp(0)
    th  = (ty2 - ty1).clamp(0)
    v   = (4 / (torch.pi ** 2)) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - d2 / c2 - alpha * v
    return (1 - ciou).mean()


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


class YOLOv8Loss(nn.Module):
    def __init__(self, num_classes, img_size=640,
                 lambda_box=7.5, lambda_cls=0.5, topk=10):
        super().__init__()
        self.num_classes = num_classes
        self.img_size    = img_size
        self.lambda_box  = lambda_box
        self.lambda_cls  = lambda_cls
        self.topk        = topk
        self.strides     = [8, 16, 32]
        self.bce         = nn.BCEWithLogitsLoss(reduction='none')

    def _build_anchors(self, device):
        all_points  = []
        all_strides = []
        for stride in self.strides:
            gs = self.img_size // stride
            gy, gx = torch.meshgrid(
                torch.arange(gs, device=device),
                torch.arange(gs, device=device),
                indexing='ij'
            )
            cx = (gx.reshape(-1).float() + 0.5) * stride
            cy = (gy.reshape(-1).float() + 0.5) * stride
            all_points.append(torch.stack([cx, cy], dim=-1))
            all_strides.append(torch.full((gs * gs,), stride, device=device, dtype=torch.float32))

        return torch.cat(all_points), torch.cat(all_strides)

    def _decode_boxes(self, raw_list, anchor_points, stride_tensor):
        reg_list = []
        cls_list = []

        for raw in raw_list:
            B = raw.shape[0]
            raw_flat = raw.permute(0, 2, 3, 1).reshape(B, -1, 4 + self.num_classes)
            reg_list.append(raw_flat[..., :4])
            cls_list.append(raw_flat[..., 4:])

        reg = torch.cat(reg_list, dim=1)
        cls = torch.cat(cls_list, dim=1)

        cx = anchor_points[:, 0] + reg[..., 0] * stride_tensor
        cy = anchor_points[:, 1] + reg[..., 1] * stride_tensor
        w  = torch.exp(reg[..., 2].clamp(-4, 4)) * stride_tensor
        h  = torch.exp(reg[..., 3].clamp(-4, 4)) * stride_tensor

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return pred_boxes, cls

    def _assign(self, pred_boxes, pred_cls, gt_boxes, gt_labels, anchor_points):
        A = pred_boxes.shape[0]
        G = gt_boxes.shape[0]
        device = pred_boxes.device

        target_boxes = torch.zeros(A, 4, device=device)
        target_cls   = torch.zeros(A, self.num_classes, device=device)
        fg_mask      = torch.zeros(A, dtype=torch.bool, device=device)

        if G == 0:
            return target_boxes, target_cls, fg_mask

        ax, ay = anchor_points[:, 0], anchor_points[:, 1]
        in_box = (
            (ax[:, None] > gt_boxes[None, :, 0]) &
            (ax[:, None] < gt_boxes[None, :, 2]) &
            (ay[:, None] > gt_boxes[None, :, 1]) &
            (ay[:, None] < gt_boxes[None, :, 3])
        )

        iou = box_iou(pred_boxes, gt_boxes)
        align_metric = iou * in_box.float()

        for g in range(G):
            metric_g = align_metric[:, g]
            topk_val, topk_idx = metric_g.topk(min(self.topk, A))

            valid = topk_val > 0
            if not valid.any():
                continue

            chosen = topk_idx[valid]
            fg_mask[chosen] = True
            target_boxes[chosen] = gt_boxes[g]
            cls_idx = gt_labels[g].long() - 1
            cls_idx = cls_idx.clamp(0, self.num_classes - 1)
            target_cls[chosen, cls_idx] = 1.0

        return target_boxes, target_cls, fg_mask

    def forward(self, raw_outputs, targets):
        device = raw_outputs[0].device
        B      = raw_outputs[0].shape[0]

        anchor_points, stride_tensor = self._build_anchors(device)
        pred_boxes, pred_cls = self._decode_boxes(raw_outputs, anchor_points, stride_tensor)

        total_box_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        num_fg         = 0

        for b in range(B):
            gt_boxes  = targets[b]['boxes'].to(device).float()
            gt_labels = targets[b]['labels'].to(device).long()

            valid = (gt_boxes[:, 2] > gt_boxes[:, 0]) & (gt_boxes[:, 3] > gt_boxes[:, 1])
            gt_boxes  = gt_boxes[valid]
            gt_labels = gt_labels[valid]

            tgt_boxes, tgt_cls, fg_mask = self._assign(
                pred_boxes[b].detach(),
                pred_cls[b].detach(),
                gt_boxes,
                gt_labels,
                anchor_points
            )

            n_fg = fg_mask.sum().item()
            num_fg += n_fg

            cls_loss = self.bce(pred_cls[b], tgt_cls)
            total_cls_loss = total_cls_loss + cls_loss.sum()

            if n_fg > 0:
                box_loss = ciou_loss(pred_boxes[b][fg_mask], tgt_boxes[fg_mask])
                total_box_loss = total_box_loss + box_loss

        normalizer = max(num_fg, 1)
        box_loss_out = self.lambda_box * total_box_loss / B
        cls_loss_out = self.lambda_cls * total_cls_loss / normalizer

        total = box_loss_out + cls_loss_out

        return total, {
            "box":   box_loss_out.item(),
            "cls":   cls_loss_out.item(),
            "total": total.item(),
        }