import torch
from iou import get_iou_matrix
from calc_offset import calc_gt_offsets


def get_req_anchors(
    anchor_boxes,
    gt_boxes,
    gt_classes,
    pos_thresh=0.7,
    neg_thresh=0.3
):

    """
    Wejście
    ----------
    anchor_boxes : (B, N, 4)
        4 = (xyxy)
        Wszystkie anchor boxy.

    gt_boxes : (B, M, 4)
        Ground truth bboxy.

    gt_classes : (B, M)
        Klasy ground truth.

    Wyjście
    -------
    positive_indices :
        (positive_batch_idx, positive_anchor_idx)

    negative_indices :
        (negative_batch_idx, negative_anchor_idx)

    GT_conf_scores : (K,)
        IoU positive anchorów.

    GT_offsets : (K, 4)
        Regression targets.

    GT_class_pos : (K,)
        Klasy przypisane positive anchorom.

    positive_anchor_coords : (K, 4)

    negative_anchor_coords : (K, 4)
    """


    # IoU matrix
    # shape: (B, N_anchors, N_gt)


    iou_matrix = get_iou_matrix(
        anchors=anchor_boxes,
        gt_bboxes=gt_boxes
    )


    # Dla każdego GT bboxa:
    # znajduje anchor o największym IoU


    max_iou_per_gt_box, _ = iou_matrix.max(
        dim=1,
        keepdim=True
    )


    # Positive anchors


    # warunek 1:
    # najlepszy anchor dla GT bboxa
    positive_anc_mask = torch.logical_and(
        iou_matrix == max_iou_per_gt_box,
        max_iou_per_gt_box > 0
    )

    # warunek 2:
    # IoU większe niż threshold
    positive_anc_mask = torch.logical_or(
        positive_anc_mask,
        iou_matrix > pos_thresh
    )


    # Każdy positive anchor może odpowiadać wielu GT bboxom.
    # Wybieramy GT bbox z największym IoU.


    positive_anchor_mask = positive_anc_mask.any(dim=-1)


    positive_batch_idx, positive_anchor_idx = torch.where(
        positive_anchor_mask
    )
    n_pos = positive_anchor_idx.shape[0]
    max_pos = min(128, n_pos)
    max_iou_per_anc, _ = iou_matrix.max(dim=-1)
    pos_ious = max_iou_per_anc[
        positive_batch_idx,
        positive_anchor_idx
    ]

    _, top_idx = torch.topk(
        pos_ious,
        k=max_pos
    )

    positive_batch_idx = positive_batch_idx[top_idx]
    positive_anchor_idx = positive_anchor_idx[top_idx]

    # dla każdego anchora:
    # index GT bboxa z największym IoU
    best_gt_idx_per_anchor = iou_matrix.argmax(dim=-1)

    positive_gt_idx = best_gt_idx_per_anchor[
        positive_batch_idx,
        positive_anchor_idx
    ]

    positive_indices = (
        positive_batch_idx,
        positive_anchor_idx
    )


    # Positive anchor coordinates


    positive_anchor_coords = anchor_boxes[
        positive_batch_idx,
        positive_anchor_idx
    ]


    # GT bboxy przypisane positive anchorom


    GT_bboxes_pos = gt_boxes[
        positive_batch_idx,
        positive_gt_idx
    ]


    # GT klasy positive anchorów


    GT_class_pos = gt_classes[
        positive_batch_idx,
        positive_gt_idx
    ]


    # IoU score positive anchorów


    

    GT_conf_scores = max_iou_per_anc[
        positive_batch_idx,
        positive_anchor_idx
    ]


    # offset gt boxes do regresji


    GT_offsets = calc_gt_offsets(
        positive_anchor_coords,
        GT_bboxes_pos
    )


    # Negative anchors


    negative_anc_mask = (max_iou_per_anc < neg_thresh)

    negative_batch_idx, negative_anchor_idx = torch.where(
        negative_anc_mask
    )


    # próbka negatives



    n_neg = negative_anchor_idx.shape[0]
    if n_pos > 0 and n_neg > 0:

        n_samples = min(n_pos, n_neg)

        rand_idx = torch.randperm(
            n_neg,
            device=anchor_boxes.device
        )[:n_samples]

        negative_batch_idx = negative_batch_idx[rand_idx]
        negative_anchor_idx = negative_anchor_idx[rand_idx]

    else:
        negative_batch_idx = torch.empty(
        0,
        dtype=torch.long,
        device=anchor_boxes.device
    )

        negative_anchor_idx = torch.empty(
        0,
        dtype=torch.long,
        device=anchor_boxes.device
    )
    negative_indices = (
        negative_batch_idx,
        negative_anchor_idx
    )
    negative_anchor_coords = anchor_boxes[
        negative_batch_idx,
        negative_anchor_idx
    ]

    return (
        positive_indices,
        negative_indices,
        GT_conf_scores,
        GT_offsets,
        GT_class_pos,
        positive_anchor_coords,
        negative_anchor_coords,
        GT_bboxes_pos
    )