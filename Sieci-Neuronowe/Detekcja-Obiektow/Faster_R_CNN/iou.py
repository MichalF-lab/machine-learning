from torchvision.ops import box_iou
import torch

def get_iou_matrix(anchors, gt_bboxes):
    # spłaszcza anchor boxy
    anchors_flat = anchors.reshape(-1, 4) # (liczba anchor, 4)
    # zwraca łączną liczbę anchor boxów
    tot_anc_boxes = anchors_flat.size(0)
    gt_bboxes = gt_bboxes.to(anchors.device).float()
    


    if gt_bboxes.numel() == 0:
        iou = torch.zeros((tot_anc_boxes, 0), device=anchors.device)
    else:
        iou = box_iou(anchors_flat, gt_bboxes)


        
    return iou