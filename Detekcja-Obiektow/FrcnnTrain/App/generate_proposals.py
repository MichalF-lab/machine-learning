import torch
from torchvision import ops

def generate_proposals(
    anchors,
    offset_pos,
    image_size
):

    image_height, image_width = image_size

    anchors = ops.box_convert(
        anchors,
        in_fmt='xyxy',
        out_fmt='cxcywh'
    )

    anc_cx = anchors[...,0]
    anc_cy = anchors[...,1]
    anc_w  = anchors[...,2]
    anc_h  = anchors[...,3]

    tx = offset_pos[...,0]
    ty = offset_pos[...,1]

    # clamp regression offsets
    tw = torch.clamp(
        offset_pos[...,2],
        min=-5,
        max=5
    )

    th = torch.clamp(
        offset_pos[...,3],
        min=-5,
        max=5
    )

    proposals = torch.zeros_like(anchors)

    # predicted center
    proposals[...,0] = anc_cx + tx * anc_w
    proposals[...,1] = anc_cy + ty * anc_h

    # predicted width / height
    proposals[...,2] = anc_w * torch.exp(tw)
    proposals[...,3] = anc_h * torch.exp(th)

    # minimal size BEFORE conversion
    proposals[...,2] = torch.clamp(
        proposals[...,2],
        min=1.0
    )

    proposals[...,3] = torch.clamp(
        proposals[...,3],
        min=1.0
    )

    proposals = ops.box_convert(
        proposals,
        in_fmt='cxcywh',
        out_fmt='xyxy'
    )

    # clamp to image boundaries
    proposals[...,0::2] = proposals[...,0::2].clamp(
        0,
        image_width - 1
    )

    proposals[...,1::2] = proposals[...,1::2].clamp(
        0,
        image_height - 1
    )

    # FINAL safety check
    x1 = proposals[...,0]
    y1 = proposals[...,1]
    x2 = proposals[...,2]
    y2 = proposals[...,3]

    x2 = torch.max(x2, x1 + 1.0)
    y2 = torch.max(y2, y1 + 1.0)

    proposals[...,0] = x1
    proposals[...,1] = y1
    proposals[...,2] = x2
    proposals[...,3] = y2

    return proposals