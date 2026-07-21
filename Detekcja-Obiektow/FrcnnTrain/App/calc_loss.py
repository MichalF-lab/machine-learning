import torch
import torch.nn.functional as F


def calc_cls_loss(conf_scores_pos,
                  conf_scores_neg):

    preds = []
    targets = []

    if conf_scores_pos.numel() > 0:
        preds.append(conf_scores_pos)
        targets.append(torch.ones_like(conf_scores_pos))

    if conf_scores_neg.numel() > 0:
        preds.append(conf_scores_neg)
        targets.append(torch.zeros_like(conf_scores_neg))

    # brak próbek
    if len(preds) == 0:
        return conf_scores_pos.sum() * 0.0

    pred = torch.cat(preds)
    target = torch.cat(targets)

    loss = F.binary_cross_entropy_with_logits(
        pred,
        target,
        reduction='sum'
    )

    loss = loss / target.size(0)

    return loss


def calc_bbox_reg_loss(
    gt_offsets,
    pred_offsets
):

    if gt_offsets.numel() == 0:
        return pred_offsets.sum() * 0.0

    loss = F.smooth_l1_loss(
        pred_offsets,
        gt_offsets,
        reduction='sum'
    )

    loss = loss / gt_offsets.size(0)

    return loss