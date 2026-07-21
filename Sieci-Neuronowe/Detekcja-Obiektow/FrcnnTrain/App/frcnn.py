
import torch.nn as nn
import torchvision
import torch
from torchvision import ops
import torch.nn.functional as F
from generate_proposals import generate_proposals
from generate_anchors import generate_anchors
from project_gt_bboxes import project_bboxes
from get_training_boxes import get_req_anchors
from calc_loss import calc_bbox_reg_loss,calc_cls_loss
from calc_offset import calc_gt_offsets

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
            #resnet.layer4
        )
        self.out_channels = 1024
    def forward(self, x):
        """
        Funkcja forward() dla backbone modelu, przyjmuje tensor postaci (B, C, H, W)
        """
        return self.backbone(x) # (B, 1024, H/16, W/16)



class ProposalModule(nn.Module):

    def __init__(
        self,
        in_features,
        image_size,
        hidden_dim=512,
        number_anchors=9,
    ):

        super().__init__()

        self.image_size = image_size

        self.conv1 = nn.Conv2d(
            in_features,
            hidden_dim,
            kernel_size=3,
            padding=1
        )

        self.conf_head = nn.Conv2d(
            hidden_dim,
            number_anchors,
            kernel_size=1
        )

        self.reg_head = nn.Conv2d(
            hidden_dim,
            number_anchors * 4,
            kernel_size=1
        )

        nn.init.constant_(
            self.conf_head.bias,
            -1.0
        )

        nn.init.normal_(
            self.reg_head.weight,
            std=0.001
        )

        nn.init.constant_(
            self.reg_head.bias,
            0
        )

    def forward(
        self,
        feature_map,
        positive_indices=None,
        negative_indices=None,
        positive_anchor_coords=None,
        negative_anchor_coords=None
    ):

        if (
            positive_indices is None or
            negative_indices is None or
            positive_anchor_coords is None or
            negative_anchor_coords is None
        ):
            mode = 'eval'
        else:
            mode = 'train'

        B = feature_map.shape[0]

        out = self.conv1(feature_map)

        out = F.relu(out)

        regression_offset_pred = self.reg_head(out)

        regression_offset_pred = regression_offset_pred.permute(
            0,2,3,1
        )

        regression_offset_pred = regression_offset_pred.reshape(
            B,-1,4
        )

        confidence_score_pred = self.conf_head(out)

        confidence_score_pred = confidence_score_pred.permute(
            0,2,3,1
        )

        confidence_score_pred = confidence_score_pred.reshape(
            B,-1
        )

        if mode == 'train':

            positive_conf_score = confidence_score_pred[
                positive_indices[0],
                positive_indices[1]
            ]

            negative_conf_score = confidence_score_pred[
                negative_indices[0],
                negative_indices[1]
            ]

            offset_pos = regression_offset_pred[
                positive_indices[0],
                positive_indices[1]
            ]

            # positive proposals
            proposals = generate_proposals(
                positive_anchor_coords,
                offset_pos,
                self.image_size
            )

            # negative proposals
            negative_offsets = torch.zeros(
                (negative_anchor_coords.shape[0], 4),
                device=negative_anchor_coords.device
            )

            negative_proposals = generate_proposals(
                negative_anchor_coords,
                negative_offsets,
                self.image_size
            )

            return (
                positive_conf_score,
                negative_conf_score,
                offset_pos,
                proposals,
                negative_proposals
            )

        else:

            return (
                confidence_score_pred,
                regression_offset_pred
            )
class RPN(nn.Module):
    def __init__(self, img_size, anchor_scales,anchor_ratios):
        super().__init__()
        
        self.img_height, self.img_width = img_size


       
        # granice dla IoU 
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3

        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.n_anc_boxes = len(self.anchor_scales) * len(self.anchor_ratios)
        self.w_conf = 1
        self.w_reg = 5

        self.feature_extractor = Backbone()
        self.proposal_module = ProposalModule(self.feature_extractor.out_channels,
    number_anchors=self.n_anc_boxes, image_size=img_size)

    def forward(self, images, gt_bboxes,gt_classes):
        feature_map = self.feature_extractor(images)

        anchors = generate_anchors(feature_map, (self.img_height,self.img_width),self.anchor_scales,self.anchor_ratios)

        gt_bboxes_proj = gt_bboxes
        positive_ind, negative_ind, GT_conf_scores, GT_offsets, GT_class_pos, \
        positive_anc_coords, negative_anc_coords, GT_bboxes_pos= get_req_anchors(anchors, gt_bboxes_proj, gt_classes)
      
        positive_conf_score, negative_conf_score, offset_pos, proposals, \
        negative_proposals = self.proposal_module(feature_map, positive_ind, \
        negative_ind, positive_anc_coords, negative_anc_coords)
        cls_loss = calc_cls_loss(positive_conf_score, negative_conf_score)
        reg_loss = calc_bbox_reg_loss(GT_offsets, offset_pos)
 
        
        total_rpn_loss = self.w_conf * cls_loss + self.w_reg * reg_loss
        
        return (
            total_rpn_loss,
            feature_map,
            proposals,
            negative_proposals,
            positive_ind,
            negative_ind,
            GT_class_pos,
            GT_bboxes_pos
        )
                
    def inference(
        self,
        images,
        conf_thresh=0.1,
        nms_thresh=0.3
    ):

        self.eval()

        pre_nms_topk = 6000
        post_nms_topk = 300

        with torch.no_grad():

            batch_size = images.size(0)

            feature_map = self.feature_extractor(images)

            anchors = generate_anchors(
                feature_map,
                (self.img_height, self.img_width),
                self.anchor_scales,
                self.anchor_ratios
            )

            conf_scores_pred, offsets_pred = self.proposal_module(
                feature_map
            )

            conf_scores_pred = conf_scores_pred.reshape(
                batch_size,
                -1
            )

            offsets_pred = offsets_pred.reshape(
                batch_size,
                -1,
                4
            )

            proposals_final = []
            conf_scores_final = []

            for i in range(batch_size):

                conf_scores = torch.sigmoid(
                    conf_scores_pred[i]
                )

                offsets = offsets_pred[i]

                # TOP-K
                num_topk = min(
                    pre_nms_topk,
                    conf_scores.size(0)
                )
                keep = conf_scores > conf_thresh

                conf_scores = conf_scores[keep]
                offsets = offsets[keep]
                anchors_img = anchors[i][keep]

                num_topk = min(
                    pre_nms_topk,
                    conf_scores.size(0)
                )

                topk_scores, topk_idx = torch.topk(
                    conf_scores,
                    num_topk
                )

                topk_offsets = offsets[topk_idx]

                topk_anchors = anchors_img[topk_idx]

                

                # decode tylko top-k
                proposals = generate_proposals(
                    topk_anchors,
                    topk_offsets,
                    (self.img_height, self.img_width)
                )
               
                # NMS
                nms_idx = ops.nms(
                    proposals,
                    topk_scores,
                    nms_thresh
                )

                nms_idx = nms_idx[:post_nms_topk]

                proposals = proposals[nms_idx]

                scores = topk_scores[nms_idx]

                proposals_final.append(
                    proposals
                )

                conf_scores_final.append(
                    scores
                )
            
            return (
                proposals_final,
                conf_scores_final,
                feature_map
            )
class ClassificationModule(nn.Module):

    def __init__(
        self,
        number_classes,
        roi_size,
        hidden_dim=512,
        out_channels=1024,
        dropout=0.3
    ):

        super().__init__()

        self.roi_size = roi_size

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(
            out_channels,
            hidden_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.cls_head = nn.Linear(
            hidden_dim,
            number_classes
        )
        self.bbox_head = nn.Linear(
            hidden_dim,
            4*number_classes
        )

    def forward(
        self,
        feature_map,
        proposals_list,
        gt_classes=None,
        gt_offsets=None
    ):

        if gt_classes is None:
            mode = 'eval'
        else:
            mode = 'train'
        total_props = sum(len(p) for p in proposals_list)

        if total_props == 0:

            if mode == 'eval':

                return (
                    torch.empty(
                        (0, self.cls_head.out_features),
                        device=feature_map.device
                    ),

                    torch.empty(
                        (
                            0,
                            self.cls_head.out_features,
                            4
                        ),
                        device=feature_map.device
                    )
                )
            zero = feature_map.sum() * 0.0

            return {
                "cls_loss": zero,
                "bbox_loss": zero,
                "total_loss": zero
            }


       
        roi_out = ops.roi_align(
            feature_map,
            proposals_list,
            output_size=self.roi_size,
            spatial_scale=1/16,
            sampling_ratio=2,
            aligned=True
        )

        # (K,C,H,W) -> (K,C,1,1)

        roi_out = self.avg_pool(roi_out)

        # (K,C)

        roi_out = roi_out.squeeze(-1).squeeze(-1)

        out = self.fc(roi_out)

        out = F.relu(out)
        out = self.dropout(out)

        cls_scores = self.cls_head(out)

        bbox_offsets = self.bbox_head(out)
        bbox_offsets = bbox_offsets.reshape(
            -1,
            self.cls_head.out_features,
            4
        )

        if mode == 'eval':
            return cls_scores, bbox_offsets

        cls_loss = F.cross_entropy(
            cls_scores,
            gt_classes.long()
        )

        # tylko foreground ROI mają bbox target
        positive_mask = gt_classes > 0

        positive_classes = gt_classes[
            positive_mask
        ].long()

        bbox_offsets_pos = bbox_offsets[
            positive_mask,
            positive_classes
        ]
        bbox_loss = calc_bbox_reg_loss(
            gt_offsets,
            bbox_offsets_pos
        )

        total_loss = cls_loss + bbox_loss

        return {
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "total_loss": total_loss
        }
    

class Faster_R_CNN(nn.Module):

    def __init__(
        self,
        img_size,
        n_classes,
        roi_size,
        anchor_scales,
        anchor_ratios
    ):

        super().__init__()

        self.rpn = RPN(
            img_size=img_size,
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios
        )

        self.classifier = ClassificationModule(
            number_classes=n_classes,
            roi_size=roi_size,
            out_channels=self.rpn.feature_extractor.out_channels
        )

    def forward(
        self,
        images,
        gt_bboxes,
        gt_classes
    ):

        (
            total_rpn_loss,
            feature_map,
            proposals,
            negative_proposals,
            positive_ind,
            negative_ind,
            GT_class_pos,
            GT_bboxes_pos

        ) = self.rpn(
            images,
            gt_bboxes,
            gt_classes
        )

        # target offsets tylko dla positive proposals
        proposal_gt_offsets = calc_gt_offsets(
            proposals.detach(),
            GT_bboxes_pos
        )

        roi_proposals_list = []
        roi_gt_classes = []
        roi_gt_offsets = []

        batch_size = images.size(0)

        for idx in range(batch_size):

            # positive idx dla obrazu
            pos_idxs = torch.where(
                positive_ind[0] == idx
            )[0]

            # negative idx dla obrazu
            neg_idxs = torch.where(
                negative_ind[0] == idx
            )[0]

            # proposals
            pos_props = proposals[
                pos_idxs
            ].detach()

            neg_props = negative_proposals[
                neg_idxs
            ].detach()

            # klasy
            pos_classes = GT_class_pos[
                pos_idxs
            ]

            # offsets tylko dla positive
            pos_offsets = proposal_gt_offsets[
                pos_idxs
            ]

            # negative klasy
            neg_classes = torch.zeros(
                len(neg_props),
                dtype=torch.long,
                device=images.device
            )

            # subsampling negatives
            if len(pos_props) == 0:

                max_neg = min(
                    32,
                    len(neg_props)
                )

            else:

                max_neg = min(
                    3 * len(pos_props),
                    len(neg_props)
                )

            if len(neg_props) > max_neg:

                rand_idx = torch.randperm(
                    len(neg_props),
                    device=images.device
                )[:max_neg]

                neg_props = neg_props[
                    rand_idx
                ]

                neg_classes = neg_classes[
                    rand_idx
                ]

            # wszystkie ROI dla classifiera
            all_props = torch.cat(
                [pos_props, neg_props],
                dim=0
            )

            all_classes = torch.cat(
                [pos_classes, neg_classes],
                dim=0
            )

            roi_proposals_list.append(
                all_props
            )

            roi_gt_classes.append(
                all_classes
            )

            # UWAGA:
            # offsets tylko dla positive ROI
            roi_gt_offsets.append(
                pos_offsets
            )

        # flatten klas
        roi_gt_classes = torch.cat(
            roi_gt_classes,
            dim=0
        )

        # flatten offsets
        if len(roi_gt_offsets) > 0:

            roi_gt_offsets = torch.cat(
                roi_gt_offsets,
                dim=0
            )
            
        else:

            roi_gt_offsets = torch.empty(
                (0, 4),
                device=images.device
            )

        classifier_output = self.classifier(
            feature_map,
            roi_proposals_list,
            roi_gt_classes,
            roi_gt_offsets
        )
        total_loss = (
            total_rpn_loss +
            classifier_output["total_loss"]
        )

        return {
            "loss":         total_loss,
            "rpn_loss":     total_rpn_loss,
            "roi_cls_loss": classifier_output["cls_loss"],
            "roi_box_loss": classifier_output["bbox_loss"],
        }

    def inference(
        self,
        images,
        conf_thresh=0.1,
        nms_thresh=0.5
    ):

        self.eval()

        batch_size = images.size(0)

        proposals_final, conf_scores_final, feature_map = self.rpn.inference(
            images,
            conf_thresh,
            nms_thresh
        )



        cls_scores, bbox_offsets = self.classifier(
            feature_map,
            proposals_final
        )
       

        cls_probs = F.softmax(
            cls_scores,
            dim=-1
        )



        scores_all, classes_all = cls_probs.max(
            dim=-1
        )

        predictions = []

        c = 0

        for i in range(batch_size):

            n_proposals = len(
                proposals_final[i]
            )

            if n_proposals == 0:

                predictions.append({
                    "boxes": torch.empty(
                        (0, 4),
                        device=images.device
                    ),
                    "scores": torch.empty(
                        (0,),
                        device=images.device
                    ),
                    "labels": torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=images.device
                    )
                })

                continue

            proposal_boxes = proposals_final[i]

            pred_classes = classes_all[
                c:c+n_proposals
            ]

            roi_offsets = bbox_offsets[
                c:c+n_proposals
            ]

            roi_idx = torch.arange(
                n_proposals,
                device=images.device
            )

            offsets_img = roi_offsets[
                roi_idx,
                pred_classes
            ]
            
            boxes = generate_proposals(
                proposal_boxes,
                offsets_img,
                (
                    self.rpn.img_height,
                    self.rpn.img_width
                )
            )

            scores = scores_all[
                c:c+n_proposals
            ]

            labels = classes_all[
                c:c+n_proposals
            ]

            # remove background + low confidence
            keep = (
                (labels > 0) &
                (scores > conf_thresh)
            )

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # safety
            if len(boxes) == 0:

                predictions.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels
                })

                c += n_proposals
                continue

            # per-class NMS
            keep = ops.batched_nms(
                boxes,
                scores,
                labels,
                nms_thresh
            )

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            predictions.append({

                "boxes": boxes,
                "scores": scores,
                "labels": labels

            })

            c += n_proposals


        return predictions