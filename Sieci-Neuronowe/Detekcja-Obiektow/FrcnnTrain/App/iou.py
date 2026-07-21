from torchvision.ops import box_iou
import torch

def get_iou_matrix(anchors, gt_bboxes):

    B,N, _ = anchors.shape
    M = gt_bboxes.shape[1]
    
    gt_bboxes = gt_bboxes.to(anchors.device).float()
    iou = torch.zeros((B, N, M), device=anchors.device)

    for b in range(B):

        valid_gt = gt_bboxes[b]

        if valid_gt.numel() > 0:
            iou[b] = box_iou(anchors[b], valid_gt)

    return iou