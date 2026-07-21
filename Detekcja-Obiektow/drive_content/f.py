from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from loaders.base_loader import get_transforms, collate_fn
from loaders.bccd_parser import BCCDDataset



def evaluate(model, val_loader, device):
    model.eval()

    metric = MeanAveragePrecision(box_format="xyxy")

    with torch.no_grad():
        for images, targets in val_loader:

            # Faster R-CNN wymaga list
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            preds = []
            gts = []

            for out, tgt in zip(outputs, targets):

                preds.append({
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu()
                })

                gts.append({
                    "boxes": tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu()
                })

            metric.update(preds, gts)

    results = metric.compute()

    return {
        "mAP": results["map"].item(),
        "mAP50": results["map_50"].item(),
        "mAP75": results["map_75"].item(),
        "precision": results["map"].item(),   # uproszczenie (torchmetrics nie daje osobno P/R per default)
        "recall": results["mar_100"].item()
    }

train_dataset = BCCDDataset(
    img_dir="data/bccd/train/img",
    annot_dir="data/bccd/train/ann",
    transforms=get_transforms(train=True)
)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)


val_dataset = BCCDDataset(
    img_dir="data/bccd/val/img",
    annot_dir="data/bccd/val/ann",
    transforms=get_transforms(train=False)
)

test_dataset = BCCDDataset(
    img_dir="data/bccd/test/img",
    annot_dir="data/bccd/test/ann",
    transforms=get_transforms(train=False)
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

num_epochs = 10


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

num_classes = 4 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)


train_loss = 0
for epoch in range(10):
    model.train()
    for images, targets in train_loader:
        
        print(1)

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
    print(f"Epoch {epoch} | Loss: {losses.item()}")
    # Walidacja
    metrics = evaluate(model, val_loader, device)

    print(f"""
    Epoch {epoch+1}
    train loss: {train_loss/len(train_loader):.4f}
    mAP: {metrics['mAP']:.4f}
    mAP50: {metrics['mAP50']:.4f}
    recall: {metrics['recall']:.4f}
    """)



   