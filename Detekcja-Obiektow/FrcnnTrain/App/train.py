import os, json, pathlib, time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import albumentations as A
from albumentations.pytorch import ToTensorV2

from frcnn import Faster_R_CNN
from base_loader import collate_fn

# ── 1. GPU info + perf knobs ─────────────────────────────────────────────────
print(f"[INFO] PyTorch     : {torch.__version__}")
print(f"[INFO] CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"[INFO] GPU         : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM        : {p.total_memory/1e9:.1f} GB")
    print(f"[INFO] Compute cap : sm_{p.major}{p.minor}")

    # Performance knobs for fixed-size inputs on modern GPUs
    torch.backends.cudnn.benchmark      = True   # autotune conv kernels
    torch.backends.cuda.matmul.allow_tf32 = True # TF32 matmul on Ampere+
    torch.backends.cudnn.allow_tf32       = True # TF32 conv on Ampere+

# ── 2. Hyperparams from env ──────────────────────────────────────────────────
MODE         = os.environ.get('MODE',           'bccd').lower()
EPOCHS       = int(os.environ.get('EPOCHS',     '30'))
BATCH        = int(os.environ.get('BATCH',      '4'))
LR           = float(os.environ.get('LR',       '5e-4'))
WORKERS      = int(os.environ.get('WORKERS',    '4'))
MOMENTUM     = float(os.environ.get('MOMENTUM',      '0.9'))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY',  '5e-4'))
GRAD_CLIP    = float(os.environ.get('GRAD_CLIP',     '10'))
EVAL_EVERY      = int(os.environ.get('EVAL_EVERY',       '1'))
UNFREEZE_AT     = int(os.environ.get('UNFREEZE_AT',      '3'))
MAX_TRAIN_HOURS = float(os.environ.get('MAX_TRAIN_HOURS', '2.0'))
SCORE_THR    = float(os.environ.get('SCORE_THRESHOLD',     '0.5'))
IOU_THR      = float(os.environ.get('IOU_THRESHOLD',       '0.5'))
MAP_SCORE_THR= float(os.environ.get('MAP_SCORE_THRESHOLD', '0.01'))
USE_AMP      = os.environ.get('USE_AMP', '1') == '1'   # FP16 mixed precision

RUNS_DIR          = pathlib.Path('/runs') / MODE
RUNS_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH   = str(RUNS_DIR / 'best_model.pth')
HISTORY_PATH      = str(RUNS_DIR / 'training_metrics.json')
TEST_METRICS_PATH = str(RUNS_DIR / 'test_metrics.json')

print(f"[INFO] MODE={MODE} | epochs={EPOCHS} batch={BATCH} lr={LR} workers={WORKERS} amp={USE_AMP}")

# ── 3. Mode-specific config ──────────────────────────────────────────────────
if MODE == 'bccd':
    from BccdYoloParser import BCCDYoloDataset
    IMG_H         = int(os.environ.get('IMG_H', '640'))
    IMG_W         = int(os.environ.get('IMG_W', '640'))
    NUM_CLASSES   = int(os.environ.get('NUM_CLASSES', '4'))   # 3 + background
    ANCHOR_SCALES = [32, 64, 128]
    ANCHOR_RATIOS = [0.5, 1.0, 2.0]
    ROI_SIZE      = (7, 7)
    CLASS_NAMES   = {1: 'RBC', 2: 'WBC', 3: 'Platelets'}
    DATA_ROOT     = pathlib.Path('/app/data/bccd')

    def _make_dataset(split, train_aug):
        return BCCDYoloDataset(
            img_dir=str(DATA_ROOT / 'images' / split),
            label_dir=str(DATA_ROOT / 'labels' / split),
            transforms=_transforms(train_aug),
        )

elif MODE == 'visdrone':
    from VisDroneYoloParser import VisDroneYoloDataset
    IMG_H         = int(os.environ.get('IMG_H', '800'))
    IMG_W         = int(os.environ.get('IMG_W', '1000'))
    NUM_CLASSES   = int(os.environ.get('NUM_CLASSES', '11'))  # 10 + background
    ANCHOR_SCALES = [8, 16, 32, 64]
    ANCHOR_RATIOS = [0.5, 1.0, 2.0]
    ROI_SIZE      = (7, 7)
    CLASS_NAMES   = {
        1: 'pedestrian', 2: 'people',    3: 'bicycle',  4: 'car',
        5: 'van',        6: 'truck',     7: 'tricycle', 8: 'awning-tricycle',
        9: 'bus',       10: 'motor',
    }
    DATA_ROOT     = pathlib.Path('/app/data/VisDrone')

    def _make_dataset(split, train_aug):
        return VisDroneYoloDataset(
            img_dir=str(DATA_ROOT / 'images' / split),
            label_dir=str(DATA_ROOT / 'labels' / split),
            transforms=_transforms(train_aug),
        )

else:
    raise ValueError(f"Unknown MODE='{MODE}'. Valid: bccd, visdrone")

# ── 4. Transforms ────────────────────────────────────────────────────────────
def _transforms(train):
    steps = [A.Resize(height=IMG_H, width=IMG_W)]
    if train:
        steps += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            A.Blur(blur_limit=3, p=0.1),
        ]
    steps += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
        ToTensorV2(),
    ]
    return A.Compose(
        steps,
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.1),
    )

# ── 5. DataLoaders ───────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pin_memory + persistent_workers + prefetch_factor → faster data pipeline
_loader_kw = dict(
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=(WORKERS > 0),
    prefetch_factor=4 if WORKERS > 0 else None,
)
train_loader = DataLoader(_make_dataset('train', True),  batch_size=BATCH, shuffle=True,  num_workers=WORKERS, **_loader_kw)
val_loader   = DataLoader(_make_dataset('val',   False), batch_size=BATCH, shuffle=False, num_workers=WORKERS, **_loader_kw)
test_loader  = DataLoader(_make_dataset('test',  False), batch_size=BATCH, shuffle=False, num_workers=WORKERS, **_loader_kw)

print(f"[INFO] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

# ── 6. Model ─────────────────────────────────────────────────────────────────
model = Faster_R_CNN(
    img_size=(IMG_H, IMG_W),
    n_classes=NUM_CLASSES,
    roi_size=ROI_SIZE,
    anchor_scales=ANCHOR_SCALES,
    anchor_ratios=ANCHOR_RATIOS,
).to(device)

for param in model.rpn.feature_extractor.parameters():
    param.requires_grad = False

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
)

# Mixed-precision scaler (no-op when disabled)
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP and torch.cuda.is_available())

# ── 7. Helpers ───────────────────────────────────────────────────────────────
def prepare_targets(targets, device):
    max_boxes  = max(max(t['boxes'].shape[0], 1) for t in targets)
    gt_bboxes  = torch.full((len(targets), max_boxes, 4), -1, device=device, dtype=torch.float32)
    gt_classes = torch.full((len(targets), max_boxes),   -1, device=device, dtype=torch.long)
    for i, t in enumerate(targets):
        n = t['boxes'].shape[0]
        if n > 0:
            gt_bboxes[i,  :n] = t['boxes'].to(device)
            gt_classes[i, :n] = t['labels'].to(device)
    return gt_bboxes, gt_classes


def _det_stats(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, score_thr, iou_thr):
    keep = pred_scores >= score_thr
    pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]
    if len(pred_boxes) == 0 and len(gt_boxes) == 0: return 0, 0, 0
    if len(pred_boxes) == 0: return 0, 0, len(gt_boxes)
    if len(gt_boxes)   == 0: return 0, len(pred_boxes), 0
    ious = box_iou(pred_boxes, gt_boxes)
    matched, tp, fp = set(), 0, 0
    for pi in torch.argsort(pred_scores, descending=True):
        best_iou, best_gi = 0.0, -1
        for gi in range(len(gt_boxes)):
            if gi in matched or pred_labels[pi] != gt_labels[gi]: continue
            v = ious[pi, gi].item()
            if v > best_iou: best_iou, best_gi = v, gi
        if best_iou >= iou_thr: tp += 1; matched.add(best_gi)
        else:                   fp += 1
    return tp, fp, len(gt_boxes) - len(matched)


def evaluate(loader):
    model.eval()
    metric      = MeanAveragePrecision(class_metrics=True, max_detection_thresholds=[1, 10, 500])
    total_tp = total_fp = total_fn = 0
    cls_stats   = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in range(1, NUM_CLASSES)}

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP and torch.cuda.is_available()):
        for images, targets in loader:
            images = torch.stack(images).to(device, non_blocking=True)
            preds  = model.inference(images)
            preds_map, tgts_map = [], []

            for pred, tgt in zip(preds, targets):
                pb = pred['boxes'].detach().cpu()
                ps = pred['scores'].detach().cpu()
                pl = pred['labels'].detach().cpu()
                gb = tgt['boxes'].detach().cpu()
                gl = tgt['labels'].detach().cpu()

                keep = ps >= MAP_SCORE_THR
                preds_map.append({'boxes': pb[keep], 'scores': ps[keep], 'labels': pl[keep]})
                tgts_map.append({'boxes': gb, 'labels': gl})

                tp, fp, fn = _det_stats(pb, pl, ps, gb, gl, SCORE_THR, IOU_THR)
                total_tp += tp; total_fp += fp; total_fn += fn

                for c in range(1, NUM_CLASSES):
                    pm, gm = pl == c, gl == c
                    tp2, fp2, fn2 = _det_stats(pb[pm], pl[pm], ps[pm], gb[gm], gl[gm], SCORE_THR, IOU_THR)
                    cls_stats[c]['tp'] += tp2
                    cls_stats[c]['fp'] += fp2
                    cls_stats[c]['fn'] += fn2

            metric.update(preds_map, tgts_map)

    res     = metric.compute()
    map_all = res['map'].item()
    map50   = res['map_50'].item()
    prec    = total_tp / (total_tp + total_fp + 1e-6)
    rec     = total_tp / (total_tp + total_fn + 1e-6)

    per_class      = {}
    map_per_class  = res.get('map_per_class')
    classes_tensor = res.get('classes')

    if map_per_class is not None and classes_tensor is not None:
        for idx, c_t in enumerate(classes_tensor):
            c  = int(c_t.item())
            s  = cls_stats[c]
            cp = s['tp'] / (s['tp'] + s['fp'] + 1e-6)
            cr = s['tp'] / (s['tp'] + s['fn'] + 1e-6)
            per_class[CLASS_NAMES.get(c, str(c))] = {
                'tp': s['tp'], 'fp': s['fp'], 'fn': s['fn'],
                'precision': cp, 'recall': cr,
                'AP': float(map_per_class[idx].item()),
            }

    return {
        'mAP': map_all, 'mAP50': map50,
        'precision': prec, 'recall': rec,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'per_class': per_class,
    }


def save_history(history):
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

# ── 8. Training loop ─────────────────────────────────────────────────────────
history     = []
best_map50  = 0.0
train_start = None          # set on first epoch, used for time-limit check

for epoch in range(EPOCHS):

    if train_start is None:
        train_start = time.time()

    if epoch == UNFREEZE_AT:
        print("[INFO] Unfreezing backbone...")
        for param in model.rpn.feature_extractor.parameters():
            param.requires_grad = True
        optimizer.add_param_group({'params': list(model.rpn.feature_extractor.parameters())})

    model.train()
    sum_loss = sum_rpn = sum_rcls = sum_rbox = 0.0
    valid = 0
    t_epoch = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:>3}/{EPOCHS}", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")
    for images, targets in pbar:
        images = torch.stack(images).to(device, non_blocking=True)
        gt_bboxes, gt_classes = prepare_targets(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=USE_AMP and torch.cuda.is_available()):
            out  = model(images, gt_bboxes, gt_classes)
            loss = out["loss"]

        if not torch.isfinite(loss):
            pbar.write(f"  [SKIP] NaN/Inf loss")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        sum_loss += loss.item()
        sum_rpn  += out["rpn_loss"].item()
        sum_rcls += out["roi_cls_loss"].item()
        sum_rbox += out["roi_box_loss"].item()
        valid    += 1

        mem = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0.0
        pbar.set_postfix(loss=f"{loss.item():.3f}", mem=f"{mem:.1f}G")

    n          = valid if valid > 0 else 1
    avg_loss   = sum_loss / n
    epoch_secs = time.time() - t_epoch
    mem_gb     = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0.0
    lr_now     = optimizer.param_groups[0]['lr']

    # ── epoch header (ultralytics-style) ────────────────────────────────────
    print(
        f"\n{'Epoch':>6}  {'GPU_mem':>7}  {'loss':>7}  {'rpn':>7}  {'roi_cls':>7}  "
        f"{'roi_box':>7}  {'lr':>8}  {'time':>6}"
    )
    print(
        f"{epoch+1:>5}/{EPOCHS:<3}  {mem_gb:>6.2f}G  {avg_loss:>7.4f}  "
        f"{sum_rpn/n:>7.4f}  {sum_rcls/n:>7.4f}  {sum_rbox/n:>7.4f}  "
        f"{lr_now:>8.2e}  {epoch_secs:>5.1f}s"
    )

    epoch_data = {
        'epoch': epoch + 1,
        'loss': avg_loss, 'rpn_loss': sum_rpn/n,
        'roi_cls_loss': sum_rcls/n, 'roi_box_loss': sum_rbox/n,
    }

    if (epoch + 1) % EVAL_EVERY == 0:
        val_m = evaluate(val_loader)
        epoch_data.update(val_m)

        # ── validation table ─────────────────────────────────────────────────
        print(
            f"\n{'':>10}{'Class':<22}{'Images':>8}{'mAP50':>8}{'mAP':>8}"
            f"{'Prec':>8}{'Recall':>8}"
        )
        n_val = len(val_loader.dataset)
        print(
            f"{'':>10}{'all':<22}{n_val:>8}{val_m['mAP50']:>8.4f}{val_m['mAP']:>8.4f}"
            f"{val_m['precision']:>8.4f}{val_m['recall']:>8.4f}"
        )
        for cls_name, cm in val_m.get('per_class', {}).items():
            print(
                f"{'':>10}{cls_name:<22}{'':>8}{cm.get('AP',0):>8.4f}{'':>8}"
                f"{cm['precision']:>8.4f}{cm['recall']:>8.4f}"
            )

        if val_m['mAP50'] >= best_map50:
            best_map50 = val_m['mAP50']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_m,
            }, BEST_MODEL_PATH)
            print(f"  [SAVE] best_model.pth  (mAP50={best_map50:.4f})")

    history.append(epoch_data)
    save_history(history)

    elapsed_h = (time.time() - train_start) / 3600
    remaining = MAX_TRAIN_HOURS - elapsed_h
    print(f"[TIME] {elapsed_h:.2f}h elapsed | {remaining:.2f}h remaining")
    if elapsed_h >= MAX_TRAIN_HOURS:
        print(f"[INFO] Time limit reached ({elapsed_h:.2f}h >= {MAX_TRAIN_HOURS}h), stopping training.")
        break

# ── 9. Test evaluation ───────────────────────────────────────────────────────
print("\n[INFO] Loading best_model.pth for final test evaluation...")
ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
test_m = evaluate(test_loader)

with open(TEST_METRICS_PATH, 'w') as f:
    json.dump(test_m, f, indent=2)

# ── 10. Summary ──────────────────────────────────────────────────────────────
print('\n' + '='*60)
print(f"[DONE] MODE={MODE} | best epoch={ckpt['epoch']+1}")
print(f"  mAP50     = {test_m['mAP50']:.4f}")
print(f"  mAP       = {test_m['mAP']:.4f}")
print(f"  precision = {test_m['precision']:.4f}")
print(f"  recall    = {test_m['recall']:.4f}")
for cls_name, m in test_m.get('per_class', {}).items():
    print(f"  {cls_name:<22}: AP={m.get('AP',0):.4f}  prec={m['precision']:.4f}  rec={m['recall']:.4f}")
print(f"\n  best_model  → {BEST_MODEL_PATH}")
print(f"  test metrics→ {TEST_METRICS_PATH}")
print('='*60)
