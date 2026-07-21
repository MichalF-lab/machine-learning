import torch
from torchvision import ops

def calc_gt_offsets(
    pos_anc_coords,
    gt_bbox_mapping
):

    pos_anc_coords = ops.box_convert(
        pos_anc_coords,
        in_fmt='xyxy',
        out_fmt='cxcywh'
    )

    gt_bbox_mapping = ops.box_convert(
        gt_bbox_mapping,
        in_fmt='xyxy',
        out_fmt='cxcywh'
    )

    gt_cx = gt_bbox_mapping[:,0]
    gt_cy = gt_bbox_mapping[:,1]
    gt_w  = gt_bbox_mapping[:,2]
    gt_h  = gt_bbox_mapping[:,3]

    anc_cx = pos_anc_coords[:,0]
    anc_cy = pos_anc_coords[:,1]
    anc_w  = pos_anc_coords[:,2]
    anc_h  = pos_anc_coords[:,3]

    eps = 1e-6

    anc_w = torch.clamp(
        anc_w,
        min=eps
    )

    anc_h = torch.clamp(
        anc_h,
        min=eps
    )

    gt_w = torch.clamp(
        gt_w,
        min=eps
    )

    gt_h = torch.clamp(
        gt_h,
        min=eps
    )

    tx_ = (gt_cx - anc_cx) / anc_w
    ty_ = (gt_cy - anc_cy) / anc_h

    tw_ = torch.log(
        gt_w / anc_w
    )

    th_ = torch.log(
        gt_h / anc_h
    )

    return torch.stack(
        [tx_, ty_, tw_, th_],
        dim=-1
    )