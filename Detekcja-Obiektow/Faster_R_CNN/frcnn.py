
import torch.nn as nn
import torchvision
import torch
from torchvision import ops
import torch.nn.functional as F
from generate_proposals import generate_proposals



class Backbone(nn.Module):
    """
    Stanowi backbone modelu Faster R-CNN, korzystając z modelu resnet50 
    """
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(weights='DEFAULT')
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.out_channels = 2048
    def forward(self, x):
        """
        Funkcja forward() dla backbone modelu, przyjmuje tensor postaci (B, C, H, W)
        """
        return self.backbone(x) # (B, 2048, H/32, W/32)


class ProposalModule(nn.Module):
    def __init__(self, in_features, hidden_dim = 512, number_anchors=9, dropout=0.3):
        super().__init__()
        self.number_anchors = number_anchors
        self.conv1 = nn.Conv2d(in_features,hidden_dim,kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.conf_head = nn.Conv2d(hidden_dim, number_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim,number_anchors*4, kernel_size=1)

    def forward(self, feature_map, positive_anchor_indices = None, negative_anchor_indices = None, positive_anchor_coords=None):
        if positive_anchor_indices is None or negative_anchor_indices is None or positive_anchor_coords is None:
            mode = 'eval'
        else:
            mode = 'train'

        out = self.conv1(feature_map)
        out = F.relu(self.dropout(out))
        regresion_offset_pred = self.reg_head(out) # (B, number_anchors*4, hmap, wmap)
        confidence_score_pred = self.conf_head(out) # (B, number_anchors, hmap, wmap)

        if mode == 'train':
            positive_conf_score = confidence_score_pred.flatten()[positive_anchor_indices]
            negative_conf_score = confidence_score_pred.flatten()[negative_anchor_indices]
            offset_pos = regresion_offset_pred.contiguous().view(-1, 4)[positive_anchor_indices]
            proposals = generate_proposals(positive_anchor_indices, offset_pos)
            return positive_conf_score, negative_conf_score, offset_pos, proposals
        elif mode=='eval':
            return positive_conf_score, regresion_offset_pred

class RPN(nn.Module):
    def __init__(self, img_size, out_size, out_channels):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.out_h, self.out_w  = out_size

        self.width_scale_factor = self.img_width // self.out_w
        self.height_scale_factor = self.img_height // self.out_h 
        self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios)
        # granice dla IoU 
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3

        self.w_conf = 1
        self.w_reg = 5

        self.feature_extractor = Backbone()
        self.proposal_module = ProposalModule(out_channels, number_anchors=self.n_anc_boxes)










class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes+1 # liczba klas w pliku + 1 dla tła

    def build_backbone(self): 
        resnet = torchvision.models.resnet50(weigths='DEFAULT')
        req_layers = list(resnet.children())[:8] # bierze pierwsze cztery warstwy, pomija elementy klasyfikacji
        backbone = nn.Sequential(*req_layers)
        return backbone
    
    def forward(self, cat):
        self.cat = cat
