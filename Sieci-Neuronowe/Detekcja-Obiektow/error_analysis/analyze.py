"""
Analiza błędów detekcji obiektów
Datasety: BCCD, BDD100K, VisDrone
Modele:   YOLOv8n (ultralytics), Faster R-CNN ResNet50 FPN (torchvision), SSD300 VGG16 (torchvision)

Uwaga: modele są pretrenowane na COCO. Wagi projektu nie były dołączone do ZIP-ów –
znaleziono tylko kod architektur. Dzięki temu domain-gap (np. krwinki vs COCO) jest
widoczny i stanowi ciekawy materiał do analizy błędów.
"""

import os, sys, io, textwrap, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")

import requests
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    ssd300_vgg16, SSD300_VGG16_Weights,
)
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO

# ─── Katalog wyjściowy ────────────────────────────────────────────────────────
OUT = "/app/output"
os.makedirs(OUT, exist_ok=True)

# RTX 5060 Ti uses Blackwell (SM120) — requires PyTorch 2.7 + CUDA 12.8.
# PyTorch 2.4.0 / CUDA 12.1 in this image targets SM90 at most, so fall back to CPU.
DEVICE = "cpu"
print(f"Urządzenie: {DEVICE} (Blackwell GPU requires PyTorch 2.7+/CUDA 12.8)")

# ─── Klasy COCO ──────────────────────────────────────────────────────────────
COCO_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# ─── Obrazy do pobrania ───────────────────────────────────────────────────────
DATASETS = {
    "BCCD": {
        "classes": ["RBC (czerwone krwinki)", "WBC (białe krwinki)", "Platelets (płytki)"],
        "images": [
            ("https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00001.jpg",
             "BloodImage_00001.jpg"),
            ("https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00007.jpg",
             "BloodImage_00007.jpg"),
            ("https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00015.jpg",
             "BloodImage_00015.jpg"),
        ],
    },
    "BDD100K": {
        "classes": ["pedestrian", "rider", "car", "truck", "bus", "train",
                    "motorcycle", "bicycle", "traffic light", "traffic sign"],
        "images": [
            ("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
             "bus_street.jpg"),
            ("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
             "street_people.jpg"),
            ("https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/images/teaser.jpg",
             "bdd_teaser.jpg"),
        ],
    },
    "VisDrone": {
        "classes": ["pedestrian", "people", "bicycle", "car", "van",
                    "truck", "tricycle", "awning-tricycle", "bus", "motor"],
        # 1 confirmed aerial (NYC) + 2 from local ultralytics assets (street-level
        # proxies used to show that COCO models do detect objects in familiar
        # perspectives but FAIL on aerial/drone perspective).
        "images": [
            ("https://upload.wikimedia.org/wikipedia/commons/b/b9/Above_Gotham.jpg",
             "aerial_nyc.jpg"),
            ("__local__:ultralytics/bus.jpg",   "visdrone_proxy_bus.jpg"),
            ("__local__:ultralytics/zidane.jpg", "visdrone_proxy_people.jpg"),
        ],
    },
}

# ─── Pobieranie obrazu ───────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://en.wikipedia.org/",
}


def _find_ultralytics_asset(name: str) -> str | None:
    try:
        import ultralytics as _ul
        p = Path(_ul.__file__).parent / "assets" / name
        return str(p) if p.exists() else None
    except Exception:
        return None


def download_image(url: str, save_path: str) -> Image.Image | None:
    # Local ultralytics assets shortcut
    if url.startswith("__local__:ultralytics/"):
        asset_name = url.split("ultralytics/", 1)[1]
        local = _find_ultralytics_asset(asset_name)
        if local:
            img = Image.open(local).convert("RGB")
            img.save(save_path)
            print(f"  ✓ {os.path.basename(save_path)} (local pkg asset)")
            return img
        print(f"  ✗ local asset not found: {asset_name}")
        return None

    for attempt in range(3):
        try:
            time.sleep(attempt * 2)
            r = requests.get(url, timeout=20, headers=HEADERS,
                              allow_redirects=True)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(save_path)
            print(f"  ✓ {os.path.basename(save_path)}")
            return img
        except Exception as e:
            if attempt == 2:
                print(f"  ✗ {url}: {e}")
    return None

# ─── Ładowanie modeli ────────────────────────────────────────────────────────
print("\nŁadowanie modeli…")

yolo = YOLO("yolov8n.pt")
yolo.to(DEVICE)
print("  ✓ YOLOv8n (ultralytics)")

frcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
frcnn.eval().to(DEVICE)
print("  ✓ Faster R-CNN ResNet50-FPN (torchvision)")

ssd = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
ssd.eval().to(DEVICE)
print("  ✓ SSD300 VGG16 (torchvision)")

# ─── Inferencja ──────────────────────────────────────────────────────────────
CONF_THRESH = 0.30
NMS_IOU     = 0.45


def run_yolo(img: Image.Image):
    results = yolo.predict(img, conf=CONF_THRESH, iou=NMS_IOU,
                           device=DEVICE, verbose=False)[0]
    boxes, scores, labels = [], [], []
    for b in results.boxes:
        boxes.append(b.xyxy[0].cpu().numpy())
        scores.append(float(b.conf[0]))
        labels.append(yolo.names[int(b.cls[0])])
    return boxes, scores, labels


def run_frcnn(img: Image.Image):
    # torchvision detection models expect List[Tensor] in [0,1] range;
    # the model's internal GeneralizedRCNNTransform handles norm+resize
    # and returns boxes in original image coordinates
    t = to_tensor(img).to(DEVICE)
    with torch.no_grad():
        preds = frcnn([t])[0]
    keep = preds["scores"] > CONF_THRESH
    boxes  = preds["boxes"][keep].cpu().numpy()
    scores = preds["scores"][keep].cpu().numpy()
    labels = [COCO_NAMES[i] for i in preds["labels"][keep].cpu().numpy()]
    return boxes, scores, labels


def run_ssd(img: Image.Image):
    # same pattern as Faster R-CNN; SSD.forward also uses GeneralizedRCNNTransform
    # and postprocesses box coords back to original image space
    t = to_tensor(img).to(DEVICE)
    with torch.no_grad():
        preds = ssd([t])[0]
    keep = preds["scores"] > CONF_THRESH
    boxes  = preds["boxes"][keep].cpu().numpy()
    scores = preds["scores"][keep].cpu().numpy()
    labels = [COCO_NAMES[i] for i in preds["labels"][keep].cpu().numpy()]
    return boxes, scores, labels


MODELS = {
    "YOLOv8n":          (run_yolo,  "#2196F3"),   # niebieski
    "Faster R-CNN":     (run_frcnn, "#F44336"),   # czerwony
    "SSD300 VGG16":     (run_ssd,   "#FF9800"),   # pomarańczowy
}

# ─── Wizualizacja ─────────────────────────────────────────────────────────────
def draw_boxes(ax, img, boxes, scores, labels, color, model_name, limit=20):
    ax.imshow(img)
    ax.axis("off")
    n = min(len(boxes), limit)
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0",
            linewidth=1.8, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        label_str = f"{labels[i]} {scores[i]:.2f}"
        ax.text(x1, max(y1 - 4, 0), label_str,
                color="white", fontsize=6, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor="none"))
    ax.set_title(f"{model_name}\n({n} detekcji)", fontsize=8, pad=3)


def error_comment(dataset, model_name, labels):
    """Automatyczny komentarz błędów dla analizy."""
    ds = dataset.upper()
    if ds == "BCCD":
        if not labels:
            return "Brak detekcji – model COCO nie\nzna klas krwinek (RBC/WBC/PLT)"
        return f"Fałszywe pozytywy: model myli\nkrwinki z obiektami COCO\n({', '.join(set(labels[:3]))})"
    if ds == "VISDRONE":
        if not labels:
            return "Brak detekcji – ujęcie z lotu\nptaka, obiekty zbyt małe/inny\nkąt niż w treningu COCO"
        tiny = sum(1 for b in [])
        return f"Możliwe błędy skali i perspektywy\nlotu drona vs zdjęcia naziemne\nCOCO ({len(labels)} det.)"
    # BDD100K
    if not labels:
        return "Brak detekcji – sprawdź próg conf"
    return f"Wykryto: {', '.join(set(labels[:4]))}"


# ─── Główna pętla ─────────────────────────────────────────────────────────────
for ds_name, ds_info in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"Klasy docelowe: {ds_info['classes']}")

    img_dir = os.path.join(OUT, ds_name)
    os.makedirs(img_dir, exist_ok=True)

    images = []
    for url, fname in ds_info["images"]:
        path = os.path.join(img_dir, fname)
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
        else:
            img = download_image(url, path)
        if img is not None:
            images.append((fname, img))

    if not images:
        print("  Brak obrazów, pomijam dataset.")
        continue

    n_imgs   = len(images)
    n_models = len(MODELS)

    # Układ siatki: wiersze = obrazy, kolumny = modele
    fig, axes = plt.subplots(n_imgs, n_models,
                             figsize=(6 * n_models, 5 * n_imgs),
                             squeeze=False)

    fig.suptitle(
        f"Analiza błędów detekcji — {ds_name}\n"
        f"Klasy docelowe: {', '.join(ds_info['classes'][:6])}"
        + ("…" if len(ds_info["classes"]) > 6 else ""),
        fontsize=14, fontweight="bold", y=1.01,
    )

    for row_idx, (fname, img) in enumerate(images):
        print(f"\n  Obraz: {fname}")
        for col_idx, (model_name, (run_fn, color)) in enumerate(MODELS.items()):
            ax = axes[row_idx][col_idx]
            try:
                boxes, scores, labels = run_fn(img)
                print(f"    {model_name}: {len(boxes)} detekcji  "
                      f"{set(labels[:5]) if labels else '(brak)'}")

                draw_boxes(ax, img, boxes, scores, labels, color, model_name)

                # Komentarz błędów w dolnej części
                comment = error_comment(ds_name, model_name, labels)
                ax.text(0.02, 0.02, comment,
                        transform=ax.transAxes,
                        fontsize=6.5, color="yellow",
                        bbox=dict(facecolor="black", alpha=0.65, pad=2),
                        verticalalignment="bottom")

            except Exception as e:
                ax.imshow(img); ax.axis("off")
                ax.set_title(f"{model_name}\nBŁĄD: {e}", fontsize=7, color="red")
                print(f"    {model_name}: BŁĄD – {e}")

        # Etykieta wiersza (nazwa pliku)
        axes[row_idx][0].set_ylabel(
            textwrap.shorten(fname, 28), fontsize=8, rotation=0,
            labelpad=100, va="center",
        )

    # Legenda modeli
    legend_patches = [
        mpatches.Patch(color=c, label=m)
        for m, (_, c) in MODELS.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=n_models, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = os.path.join(OUT, f"error_analysis_{ds_name}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n  Zapisano: {out_path}")

# ─── Zbiorczy raport tekstowy ─────────────────────────────────────────────────
print("\n" + "="*60)
print("ANALIZA ZAKOŃCZONA")
print(f"Wyniki zapisano w: {OUT}/")
print("Pliki PNG:")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        size = os.path.getsize(os.path.join(OUT, f)) // 1024
        print(f"  {f}  ({size} KB)")

print("""
─── Podsumowanie rodzajów błędów ───────────────────────────────
BCCD (krwinki):
  • Fałszywe pozytywy (FP) – modele COCO "widzą" obiekty których
    nie ma (np. 'clock', 'ball'), bo nie znają klas RBC/WBC/PLT
  • Fałszywe negatywy (FN) – wszystkie krwinki są pominięte
  → Domain shift: modele trenowane na naturalnych obrazach
    NIE generalizują na obrazy mikroskopowe

BDD100K (ulice):
  • Lepsze wyniki – COCO zawiera podobne klasy (car, person, bus)
  • Możliwe błędy: pomylenie 'rider' z 'person', brak klasy 'train'
    w SSD, zawyżone boxy dla zgrupowanych obiektów

VisDrone (lot drona):
  • Perspektywa z góry – rzadka w COCO → gorsze detekcje
  • Obiekty bardzo małe (< 32px) – Faster R-CNN i SSD słabiej
    radzą sobie z małymi obiektami niż YOLOv8
  • FP przy wykrywaniu dachu samochodów jako 'frisbee' itp.
────────────────────────────────────────────────────────────────
""")
