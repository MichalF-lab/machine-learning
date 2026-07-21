import torch

def project_bboxes(gt_bboxes, width_scale_factor, height_scale_factor, mode='f2p'):
    assert mode in ['f2p', 'p2f']
    

    proj_bboxes = gt_bboxes.clone().float()

    invalid_bbox_mask = (proj_bboxes < 0)

    if mode == 'f2p': 
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor

    proj_bboxes.masked_fill_(invalid_bbox_mask, -1)



    return proj_bboxes