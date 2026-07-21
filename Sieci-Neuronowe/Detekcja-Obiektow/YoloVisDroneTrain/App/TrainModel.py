import os
import sys
import shutil
import time
import pathlib

# ── 1. GPU verification ──────────────────────────────────────────────────────
import torch

print(f"[INFO] PyTorch     : {torch.__version__}")
print(f"[INFO] CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    print(f"[INFO] GPU         : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM        : {vram_gb:.1f} GB")
    print(f"[INFO] Compute cap : sm_{props.major}{props.minor}")
else:
    print("[WARN] No CUDA device detected — training will run on CPU (extremely slow!).")

# ── 2. Stage dataset to fast container fs (ext4) ────────────────────────────
# Windows bind-mounts go through WSL2 9P protocol → slow random reads of
# small files. Copying the dataset once to /tmp (native ext4 in WSL2) makes
# data loading 5–10× faster. Cost: ~10–30s one-time copy per container start.
VOLUME_DATA_DIR = pathlib.Path('/app/datasets')    # bind-mount → ./Datasets on host
FAST_DATA_DIR   = pathlib.Path('/tmp/datasets')    # container internal (fast ext4)
DATASET_NAME    = 'VisDrone'


def stage_dataset() -> pathlib.Path:
    """Copy VisDrone from slow bind-mount to fast container fs. Returns active datasets dir."""
    src = VOLUME_DATA_DIR / DATASET_NAME
    dst = FAST_DATA_DIR / DATASET_NAME

    if not src.exists():
        print(f"[INFO] Dataset not on volume yet — ultralytics will download to {VOLUME_DATA_DIR}")
        print(f"[INFO] After first download, future runs will stage to fast fs automatically")
        return VOLUME_DATA_DIR

    if dst.exists():
        print(f"[INFO] Dataset already staged at {dst} — reusing")
        return FAST_DATA_DIR

    # Measure source size
    try:
        size_bytes = sum(f.stat().st_size for f in src.rglob('*') if f.is_file())
        size_gb = size_bytes / 1e9
    except Exception:
        size_gb = 0.0

    print(f"[INFO] Staging dataset ({size_gb:.2f} GB): {src} → {dst}")
    t0 = time.time()
    FAST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    elapsed = time.time() - t0
    rate_mbps = (size_gb * 1024) / max(elapsed, 0.01)
    print(f"[INFO] Staging done in {elapsed:.1f}s ({rate_mbps:.0f} MB/s)")
    return FAST_DATA_DIR


active_data_dir = stage_dataset()

# ── 3. Configure ultralytics to use active dataset directory ────────────────
from ultralytics import settings as yolo_settings
yolo_settings.update({'datasets_dir': str(active_data_dir)})

import ultralytics
print(f"[INFO] ultralytics : {ultralytics.__version__}")
print(f"[INFO] datasets_dir: {active_data_dir}")

from ultralytics import YOLO

# ── 4. Hyperparameters from environment variables ───────────────────────────
EPOCHS      = int(os.environ.get('EPOCHS',    '100'))
BATCH       = int(os.environ.get('BATCH',      '32'))
IMGSZ       = int(os.environ.get('IMGSZ',     '640'))
DEVICE      = os.environ.get('DEVICE',          '0')
RESUME      = os.environ.get('RESUME',     'false').lower() == 'true'
RUN_NAME    = os.environ.get('RUN_NAME',    'Train')
PROJECT_DIR = '/runs'
WORKERS     = int(os.environ.get('WORKERS',     '8'))

# CACHE: 'false'→False, 'true'/'ram'→'ram', 'disk'→'disk'
_cache_raw  = os.environ.get('CACHE', 'false').lower()
CACHE = {'false': False, 'true': 'ram', 'ram': 'ram', 'disk': 'disk'}.get(_cache_raw, False)

print(f"[INFO] epochs={EPOCHS}  batch={BATCH}  imgsz={IMGSZ}  "
      f"device={DEVICE}  resume={RESUME}  run={RUN_NAME}")
print(f"[INFO] workers={WORKERS}  cache={CACHE}")

# ── 5. Load model ───────────────────────────────────────────────────────────
LAST_PT = pathlib.Path(PROJECT_DIR) / RUN_NAME / 'weights' / 'last.pt'

if RESUME and LAST_PT.exists():
    print(f"[INFO] Resuming from {LAST_PT}")
    model = YOLO(str(LAST_PT))
else:
    if RESUME:
        print(f"[WARN] RESUME=true but {LAST_PT} not found — starting from yolov8n.pt")
    model = YOLO('yolov8n.pt')

# ── 6. Train ────────────────────────────────────────────────────────────────
results = model.train(
    data='VisDrone.yaml',

    # basic
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMGSZ,
    device=DEVICE,

    # optimiser & learning rate
    optimizer='auto',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    # warmup
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.0,

    # loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,

    # augmentation (YOLOv8 defaults)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=10,
    auto_augment='randaugment',
    erasing=0.4,

    # data loading — staged dataset on fast ext4 → workers can actually feed GPU
    workers=WORKERS,
    cache=CACHE,        # default False — staging already gives us fast I/O

    # training options
    amp=True,
    seed=0,
    deterministic=True,
    patience=20,

    # output
    save=True,
    save_period=10,
    plots=True,
    val=True,

    # resume / project
    resume=RESUME,
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
)

# ── 7. Final summary ────────────────────────────────────────────────────────
best_pt = pathlib.Path(PROJECT_DIR) / RUN_NAME / 'weights' / 'best.pt'
last_pt = pathlib.Path(PROJECT_DIR) / RUN_NAME / 'weights' / 'last.pt'

print('\n' + '=' * 60)
print('[DONE] Training finished')
print(f'  best.pt  → {best_pt}')
print(f'  last.pt  → {last_pt}')
try:
    metrics = results.results_dict
    for key in [
        'metrics/precision(B)',
        'metrics/recall(B)',
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)',
    ]:
        val = metrics.get(key)
        if val is not None:
            print(f'  {key:<30}: {val:.4f}')
except Exception as exc:
    print(f'  (metrics extraction failed: {exc})')
print('=' * 60)
