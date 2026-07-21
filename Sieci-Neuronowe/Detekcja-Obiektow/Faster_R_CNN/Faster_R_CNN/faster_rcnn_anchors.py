import torch
from torchvision import ops
import math

def generate_anchors(feature_map, image_size, anchor_scales,anchor_ratios, stride=None):

    device = feature_map.device
    _,_,H,W, = feature_map.shape
    if stride is None:
        stride_y = image_size[0] / H
        stride_x = image_size[1] / W                            
    else:
        stride_x = stride
        stride_y = stride

    center_x = (torch.arange(W, device=device)+.5) * stride_x
    center_y = (torch.arange(H, device=device)+.5) * stride_y

    center_y, center_x = torch.meshgrid(center_y, center_x, indexing="ij")
    num_anchors = len(anchor_scales) * len(anchor_ratios) # tworzymy 9 anchor boxów dla każdego anchora
    anchors = torch.zeros((H, W, num_anchors, 4), device=device)


    for i, scale in enumerate(anchor_scales):
        for j, ratio in enumerate(anchor_ratios):
            index = i * len(anchor_ratios) + j
            w = scale * math.sqrt(ratio)
            h = scale/math.sqrt(ratio)
            xmin = center_x - w / 2
            ymin = center_y - h / 2
            xmax = center_x + w / 2
            ymax = center_y + h / 2
           
            anchors[:, :, index, 0] = xmin
            anchors[:, :, index, 1] = ymin
            anchors[:, :, index, 2] = xmax
            anchors[:, :, index, 3] = ymax

    
    anchors = ops.clip_boxes_to_image(anchors, size=image_size)

    return anchors

