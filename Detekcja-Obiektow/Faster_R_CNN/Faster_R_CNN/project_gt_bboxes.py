import torch

def project_bboxes(gt_bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
    assert mode in ['a2p', 'p2a']
    
    batch_size = gt_bboxes.size(0)
    proj_bboxes = gt_bboxes.clone().reshape(batch_size, -1, 4)

    invalid_bbox_mask = (proj_bboxes < 0)

    if mode == 'a2p':
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor

    proj_bboxes.masked_fill_(invalid_bbox_mask, -1)

    proj_bboxes = proj_bboxes.view_as(gt_bboxes)

    return proj_bboxes