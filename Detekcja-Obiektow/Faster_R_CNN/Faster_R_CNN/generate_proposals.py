import torch
from torchvision import ops

def generate_proposals(anchors, offset_pos):
    # zmienia format z xyxy na center_xcenter_ywh
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')
    proposals = torch.zeros_like(anchors)
    proposals[:,0] = anchors[:, 0] + offset_pos[:, 0]*anchors[:,2]
    proposals[:,1] = anchors[:, 1] + offset_pos[:, 1]*anchors[:,3]
    proposals[:,2] = anchors[:, 2]*torch.exp(offset_pos[:,2])
    proposals[:,3] = anchors[:, 3]*torch.exp(offset_pos[:,3])
    # powraca z formatem do xyxy
    proposals = ops.box_convert(proposals, in_fmt='cxcywh', out_fmt='xyxy')
    return proposals