import json
import time
import torch
from torch.utils.data import DataLoader

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import albumentations as A
from albumentations.pytorch import ToTensorV2

from frcnn import Faster_R_CNN
from base_loader import collate_fn
from bccd_parser import BCCDDataset


# =========================================================
# CONFIG
# =========================================================

CONFIG = {

    # training
    "num_epochs": 1,
    "batch_size": 2,
    "learning_rate": 5e-4,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "anchor_scales": [32,64,128],
    "anchor_ratios": [0.5, 1.0, 2],


    # evaluation
    "eval_every": 1,
    "score_threshold": 0.5,
    "iou_threshold": 0.5,
    "stats_score_threshold": 0.3,
    "map_score_threshold": 0.01,
    "test_path": "test_metrics.json",

    # model
    "img_size": (640, 640),
    "roi_size": (7, 7),
    "num_classes": 4, # klasy w dataset i trzeba dodać +1 na background

    # checkpointing
    "best_model_path": "best_model.pth",
    "history_path": "training_metrics.json",

    # optimization
    "grad_clip": 10
}
### RESNET50 Mean, std



def get_transforms(train=True):

    transforms = [
        A.Resize(
            height=640,
            width=640
        )
    ]

    # =====================================================
    # TRAIN AUGMENTATIONS
    # =====================================================

    if train:

        transforms.extend([

            A.HorizontalFlip(p=0.5),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.2
            ),

            A.Blur(
                blur_limit=3,
                p=0.1
            )
        ])

    # =====================================================
    # IMAGENET NORMALIZATION (ResNet50)
    # =====================================================

    transforms.extend([

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=1.0
        ),

        ToTensorV2()
    ])

    return A.Compose(

        transforms,

        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.1
        )
    )

# =========================================================
# DATASETS
# =========================================================

def create_dataloaders(batch_size=CONFIG["batch_size"]):

    train_dataset = BCCDDataset(
        img_dir="data/bccd/train/img",
        annot_dir="data/bccd/train/ann",
        transforms=get_transforms(train=True)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataset = BCCDDataset(
        img_dir="data/bccd/test/img",
        annot_dir="data/bccd/test/ann",
        transforms=get_transforms(train=False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    val_dataset = BCCDDataset(
        img_dir="data/bccd/val/img",
        annot_dir="data/bccd/val/ann",
        transforms=get_transforms(train=False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader
# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# =========================================================
# MODEL
# =========================================================

def build_model(device):

    model = Faster_R_CNN(
        img_size=CONFIG["img_size"],
        n_classes=CONFIG["num_classes"], # RBC, WBC, Platelets, background
        roi_size=CONFIG["roi_size"],
        anchor_scales=CONFIG["anchor_scales"],
        anchor_ratios=CONFIG["anchor_ratios"]
    )

    model.to(device)

    return model


# =========================================================
# FREEZE BACKBONE (optional)
# =========================================================
def freeze_backbone(model):

    for param in model.rpn.feature_extractor.parameters():
        param.requires_grad = False
# =========================================================
# OPTIMIZER
# =========================================================
def unfreeze_backbone(model):

    for param in model.rpn.feature_extractor.parameters():
        param.requires_grad = True
def build_optimizer(
    model,
    lr=CONFIG["learning_rate"],
    momentum=CONFIG["momentum"],
    weight_decay=CONFIG["weight_decay"]
):

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    return optimizer
# =========================================================
# METRIC HELPERS
# =========================================================

def compute_detection_stats(
    pred_boxes,
    pred_labels,
    pred_scores,
    gt_boxes,
    gt_labels,
    score_threshold,
    iou_threshold=CONFIG["iou_threshold"],
    
):
    """
    Computes TP, FP, FN for object detection using:
    - IoU matching
    - class matching
    - one-to-one assignment
    """

    # =====================================================
    # FILTER BY SCORE
    # =====================================================

    keep = pred_scores >= score_threshold

    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    # =====================================================
    # EDGE CASES
    # =====================================================

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    # =====================================================
    # IOU MATRIX
    # =====================================================

    ious = box_iou(pred_boxes, gt_boxes)

    matched_gt = set()

    tp = 0
    fp = 0

    # =====================================================
    # SORT PREDICTIONS BY SCORE DESC
    # =====================================================

    sorted_indices = torch.argsort(
        pred_scores,
        descending=True
    )

    # =====================================================
    # MATCHING
    # =====================================================

    for pred_idx in sorted_indices:

        pred_label = pred_labels[pred_idx]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx in range(len(gt_boxes)):

            if gt_idx in matched_gt:
                continue

            gt_label = gt_labels[gt_idx]

            # class must match
            if pred_label != gt_label:
                continue

            iou = ious[pred_idx, gt_idx].item()

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:

            tp += 1
            matched_gt.add(best_gt_idx)

        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn



def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    num_epochs=CONFIG["num_epochs"]
):

    model.train()

    epoch_loss = 0.0
    valid_batches = 0

    for batch_idx, (images, targets) in enumerate(loader):

        

        # =====================================================
        # IMAGES
        # =====================================================

        images = torch.stack(images).to(device)

        gt_bboxes, gt_classes = prepare_targets(
            targets,
            device
        )

        # =====================================================
        # FORWARD
        # =====================================================

        loss = model(
            images,
            gt_bboxes,
            gt_classes
        )

        # skip nan / inf losses
        if not torch.isfinite(loss):

            print(
                f"Skipping invalid loss: {loss.item()}"
            )

            continue

        # =====================================================
        # BACKWARD
        # =====================================================

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=CONFIG["grad_clip"]
        )

        optimizer.step()

        # =====================================================
        # LOGGING
        # =====================================================

        epoch_loss += loss.item()
        valid_batches += 1

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Batch [{batch_idx+1}/{len(loader)}] "
            f"Loss: {loss.item():.4f}"
        )

    # =====================================================
    # SAFE AVERAGE
    # =====================================================

    if valid_batches == 0:

        avg_loss = float("inf")

    else:

        avg_loss = epoch_loss / valid_batches

    return {
        "loss": avg_loss
    }
CLASS_NAMES = {
    1: "RBC",
    2: "WBC",
    3: "Platelets"
}


def evaluate(
    model,
    loader,
    device,
    iou_threshold=CONFIG["iou_threshold"],
    score_threshold=CONFIG["stats_score_threshold"]
):

    model.eval()

    num_classes = CONFIG["num_classes"]

    # =====================================================
    # TORCHMETRICS mAP
    # =====================================================

    metric = MeanAveragePrecision(
        class_metrics=True
    )

    # =====================================================
    # GLOBAL STATS
    # =====================================================

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # =====================================================
    # PER CLASS STATS
    # =====================================================

    class_stats = {
        cls_id: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
        for cls_id in range(1, num_classes)
    }

    # =====================================================
    # EVAL LOOP
    # =====================================================

    with torch.no_grad():

        for images, targets in loader:

            images = torch.stack(images).to(device)

            predictions = model.inference(images)

            preds_for_map = []
            targets_for_map = []

            for pred, target in zip(predictions, targets):

                # =========================================
                # PREDICTIONS
                # =========================================

                pred_boxes = pred["boxes"].detach().cpu()
                pred_scores = pred["scores"].detach().cpu()
                pred_labels = pred["labels"].detach().cpu()

                # =========================================
                # GT
                # =========================================

                gt_boxes = target["boxes"].detach().cpu()
                gt_labels = target["labels"].detach().cpu()

                # =========================================
                # mAP FILTER
                # =========================================

                map_keep = (
                    pred_scores >= CONFIG["map_score_threshold"]
                )

                preds_for_map.append({
                    "boxes": pred_boxes[map_keep],
                    "scores": pred_scores[map_keep],
                    "labels": pred_labels[map_keep]
                })

                targets_for_map.append({
                    "boxes": gt_boxes,
                    "labels": gt_labels
                })

                # =========================================
                # GLOBAL TP / FP / FN
                # =========================================

                tp, fp, fn = compute_detection_stats(
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn

                # =========================================
                # PER CLASS TP / FP / FN
                # =========================================

                for class_id in range(1, num_classes):

                    pred_mask = pred_labels == class_id
                    gt_mask = gt_labels == class_id

                    tp, fp, fn = compute_detection_stats(
                        pred_boxes=pred_boxes[pred_mask],
                        pred_labels=pred_labels[pred_mask],
                        pred_scores=pred_scores[pred_mask],
                        gt_boxes=gt_boxes[gt_mask],
                        gt_labels=gt_labels[gt_mask],
                        score_threshold=score_threshold,
                        iou_threshold=iou_threshold
                    )

                    class_stats[class_id]["tp"] += tp
                    class_stats[class_id]["fp"] += fp
                    class_stats[class_id]["fn"] += fn

            # =============================================
            # UPDATE MAP
            # =============================================

            metric.update(
                preds_for_map,
                targets_for_map
            )

    # =====================================================
    # FINAL mAP
    # =====================================================

    results = metric.compute()

    map_all = results["map"].item()
    map50 = results["map_50"].item()

    # =====================================================
    # GLOBAL PRECISION / RECALL
    # =====================================================

    precision = total_tp / (
        total_tp + total_fp + 1e-6
    )

    recall = total_tp / (
        total_tp + total_fn + 1e-6
    )

    # =====================================================
    # PER CLASS METRICS
    # =====================================================

    per_class_metrics = {}

    map_per_class = results.get("map_per_class", None)

    for class_id, stats in class_stats.items():

        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        class_precision = tp / (
            tp + fp + 1e-6
        )

        class_recall = tp / (
            tp + fn + 1e-6
        )

        class_name = CLASS_NAMES.get(
            class_id,
            str(class_id)
        )

        class_result = {
            "precision": class_precision,
            "recall": class_recall,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

        # =============================================
        # PER CLASS AP
        # =============================================

        if map_per_class is not None:

            class_result["AP"] = (
                map_per_class[class_id - 1]
                .item()
            )

        per_class_metrics[class_name] = class_result

    # =====================================================
    # RETURN
    # =====================================================

    return {
        "mAP": map_all,
        "mAP50": map50,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": per_class_metrics
    }

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=CONFIG["num_epochs"],
    eval_every=CONFIG["eval_every"]
):

    history = []

    best_map50 = -1.0

    for epoch in range(num_epochs):
        if epoch == 3:


            print("Unfreezing backbone...")
            unfreeze_backbone(model)

        # =====================================================
        # TRAIN
        # =====================================================

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs
        )

        
        # =====================================================
        # EPOCH DATA
        # =====================================================

        epoch_data = {
            "epoch": epoch + 1,
            **train_metrics
        }

        # =====================================================
        # EVAL
        # =====================================================

        if (epoch + 1) % eval_every == 0:

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device
            )

            epoch_data.update(val_metrics)

            print(val_metrics)

            # ================================================
            # SAVE BEST
            # ================================================

            if val_metrics["mAP50"] >= best_map50:

                best_map50 = val_metrics["mAP50"]

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    path=CONFIG["best_model_path"]
                )

        # =====================================================
        # HISTORY
        # =====================================================

        history.append(epoch_data)


        save_history(
            history,
            CONFIG["history_path"]
        )

    return history

def prepare_targets(targets, device):

    max_boxes = max(
        max(t["boxes"].shape[0], 1)
        for t in targets
    )

    gt_bboxes = torch.full(
        (len(targets), max_boxes, 4),
        -1,
        device=device,
        dtype=torch.float32
    )

    gt_classes = torch.full(
        (len(targets), max_boxes),
        -1,
        device=device,
        dtype=torch.long
    )

    for i, t in enumerate(targets):

        n = t["boxes"].shape[0]

        if n > 0:

            gt_bboxes[i, :n] = t["boxes"].to(device)

            gt_classes[i, :n] = t["labels"].to(device)

    return gt_bboxes, gt_classes
def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    path
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }

    torch.save(
        checkpoint,
        path
    )
    
def save_history(history, path):

    with open(path, "w") as f:
        json.dump(
            history,
            f,
            indent=4
        )

# =========================================================
# MAIN
# =========================================================
def test(
    model,
    test_loader,
    device,
    checkpoint_path=CONFIG["best_model_path"]
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"Loaded checkpoint from: {checkpoint_path}")

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device
    )

    print("\n========== TEST RESULTS ==========")

    for key, value in test_metrics.items():

        print(f"{key}: {value}")
    with open(CONFIG["test_path"], "w") as f:

        json.dump(
            test_metrics,
            f,
            indent=4
        )

    return test_metrics
def main():

    train_loader, val_loader, test_loader = create_dataloaders()

    model = build_model(device)

    freeze_backbone(model)

    optimizer = build_optimizer(model)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device
    )

    test_metrics = test(
        model=model,
        test_loader=test_loader,
        device=device
    )

    return history, test_metrics


if __name__ == "__main__":
    main()