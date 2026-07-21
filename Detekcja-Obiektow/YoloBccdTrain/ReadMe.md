# YOLOv8n — Trening na BCCD (Blood Cell Detection)

YOLOv8n trenowany na **BCCD** (Blood Cell Count and Detection) — 3 klasy komórek krwi.
Pełny pipeline w Dockerze z GPU, identyczny jak `YoloVisDroneTrain`.

`ultralytics==8.4.48` · `imgsz=640` · 3 klasy · 100 epok (domyślnie)

---

## Klasy

| ID | Nazwa     | Opis                                |
|----|-----------|-------------------------------------|
| 0  | Platelets | Płytki krwi (małe, liczne)          |
| 1  | RBC       | Erytrocyty / czerwone krwinki       |
| 2  | WBC       | Leukocyty / białe krwinki (większe) |

---

## Dataset

Pobierany **automatycznie** z [Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset)
(format Pascal VOC, ~7 MB).

`App/BccdSetup.py`:
1. Pobiera ZIP z GitHuba (~7 MB, kilka sekund)
2. Wypakowuje
3. Konwertuje anotacje XML (Pascal VOC) → YOLO `.txt`
4. Tworzy podział train/val/test wg dołączonych plików `ImageSets/Main/*.txt`
5. Zapisuje do `Datasets/BCCD/` na wolumenie (jednorazowo)

Po pierwszym uruchomieniu kolejne starty pomijają pobieranie i konwersję.

---

## Struktura

```
YoloBccdTrain/
├── Dockerfile             — identyczny jak VisDrone (pytorch 2.7 + cuda 12.8 + ultralytics 8.4.48)
├── DockerCompose.yml      — BCCD-specific hyperparams
├── App/
│   ├── TrainModel.py      — orchestrates setup + training
│   └── BccdSetup.py       — VOC→YOLO converter + downloader
├── Datasets/              — wolumen: BCCD images + YOLO labels (po pierwszym setupie)
├── Runs/                  — wolumen: wagi, wykresy, metryki
└── ReadMe.md
```

---

## Uruchomienie

```powershell
cd D:\CLCO\YoloBccdTrain
docker compose -f DockerCompose.yml up --build
```

Pierwszy run: build obrazu (Docker layers głównie z cache jeśli VisDrone był budowany), pobranie BCCD (~kilka sekund), trening 100 epok.

Przy 360 obrazach i `batch=16` epoka trwa kilka–kilkanaście sekund — **cały trening ~10–15 min** na RTX 5060 Ti.

---

## Domyślne hyperparams (DockerCompose.yml)

| Param   | Wartość  | Komentarz                                                   |
|---------|----------|-------------------------------------------------------------|
| EPOCHS  | 100      | dużo dla małego datasetu, ale early-stopping odetnie nadmiar |
| BATCH   | 16       | ~19 batchy/epokę = dobra częstotliwość update'ów            |
| IMGSZ   | 640      | standard YOLOv8                                              |
| WORKERS | 4        | mały dataset, więcej workerów = niepotrzebne                |
| CACHE   | ram      | 360 obrazów × 640×640 ≈ 1.5 GB w RAM → idealnie się mieści  |

---

## Wyniki

Po treningu:

```
Runs/Train/
├── weights/
│   ├── best.pt        — najlepszy mAP50 na walidacji
│   └── last.pt        — ostatnia epoka
├── results.png        — krzywe loss / precision / recall / mAP
├── confusion_matrix.png
├── PR_curve.png
└── results.csv        — metryki per-epoka
```

Realistyczne mAP50 dla BCCD przy YOLOv8n: **0.85–0.95** (zbiór jest "łatwy" w porównaniu do VisDrone).

---

## CUDA out of memory

Mało prawdopodobne przy `batch=16` i imgsz=640 (~3–4 GB VRAM). Gdyby się zdarzyło:

```yaml
- BATCH=8
```

---

## Zmiana hyperparams

W `DockerCompose.yml` → sekcja `environment`. Bez restartu kodu Pythona.

---

## Wznowienie przerwanego treningu

```yaml
- RESUME=true
```

i ponowny `docker compose -f DockerCompose.yml up`.
