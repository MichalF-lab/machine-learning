import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from frcnn import Backbone
from faster_rcnn_anchors import generate_anchors
from PIL import Image
import torchvision.transforms as T
from torchvision.ops import box_iou
from iou_matrix import get_iou_matrix

img = Image.open("../bccd/train/img/BloodImage_00001.jpeg").convert("RGB")

transform = T.Compose([
    T.Resize((416, 416)),
    T.ToTensor()
])



x = transform(img).unsqueeze(0)  # [1,3,416,416]

model = Backbone()

anchor_scales = [16,32,64]
anchor_ratios = [0.5, 1.0, 2.0]

y = model(x)

anchors = generate_anchors(y, (416,416), anchor_scales, anchor_ratios)

print(anchors.shape)


def show_anchors(image, anchors, max_anchors=20000):

    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1,2,0).cpu().numpy())

    anchors = anchors.view(-1, 4)

    for i in range(min(len(anchors), max_anchors)):
        x1, y1, x2, y2 = anchors[i]

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=0.5,
            edgecolor='b',
            facecolor='none',
            alpha=0.3
        )
        ax.add_patch(rect)

    plt.show()


show_anchors(x[0], anchors)
