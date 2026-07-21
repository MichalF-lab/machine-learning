# YOLOv8n — Trening na VisDrone

YOLOv8n (nano) trenowany na VisDrone2019-DET w Dockerze z GPU.  
`ultralytics==8.4.48` · `imgsz=640` · 10 klas · 100 epok (domyślnie)

---

## Struktura katalogów

```
YoloVisDroneTrain/
├── Dockerfile            — obraz: PyTorch 2.7 + CUDA 12.8 + ultralytics 8.4.48
├── DockerCompose.yml     — serwis Train z dostępem do GPU i wolumenami
├── App/
│   └── TrainModel.py     — skrypt treningowy (parametry z ENV)
├── Datasets/             — wolumin: pobrany i skonwertowany VisDrone (~2.3 GB)
├── Runs/                 — wolumin: wagi, wykresy, logi
└── ReadMe.md
```

---

## Wymagania

- Docker Desktop z włączonym backendem WSL2
- NVIDIA Container Toolkit (sterownik GPU ≥ 530, CUDA ≥ 12.8 na hoście)
- RTX 5060 Ti (lub inny Blackwell/Ampere/Ada)

Weryfikacja GPU poza Dockerem:
```powershell
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

---

## Zbudowanie obrazu i uruchomienie treningu

```powershell
cd D:\CLCO\YoloVisDroneTrain

# Buduje obraz + pobiera dataset + trenuje (może potrwać kilka–kilkanaście godzin)
docker compose -f DockerCompose.yml up --build
```

Pierwsze uruchomienie pobiera `yolov8n.pt` (~6 MB) i dataset VisDrone (~2.3 GB) do `Datasets/`.  
Kolejne uruchomienia korzystają z cache — nie pobierają danych ponownie.

### Szybki smoke-test (3 epoki, ~15 minut)

Przed pełnym treningiem sprawdź, że GPU + dataset + zapis wyników działają:

1. Otwórz `DockerCompose.yml`
2. Zmień `EPOCHS=100` → `EPOCHS=3`
3. Uruchom `docker compose -f DockerCompose.yml up --build`
4. Po sukcesie przywróć `EPOCHS=100` i uruchom ponownie

---

## Gdzie szukać wyników

Po zakończeniu treningu wszystkie pliki są w `Runs/Train/`:

```
Runs/Train/
├── weights/
│   ├── best.pt        — model z najlepszym mAP50 na zbiorze val
│   └── last.pt        — model z ostatniej epoki (lub przerwanego treningu)
├── results.png        — krzywe loss, precision, recall, mAP
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
└── results.csv        — metryki per-epoka
```

Finalne metryki są też wypisane na końcu logów kontenera.

---

## Jak zmienić liczbę epok lub inne parametry

Edytuj sekcję `environment` w `DockerCompose.yml`:

```yaml
environment:
  - EPOCHS=50      # liczba epok
  - BATCH=8        # rozmiar batcha
  - IMGSZ=640      # rozmiar wejścia
  - DEVICE=0       # indeks GPU
  - RUN_NAME=Train # nazwa folderu w Runs/
```

Nie musisz dotykać `TrainModel.py`.

---

## Wznowienie przerwanego treningu

1. W `DockerCompose.yml` ustaw `RESUME=true`
2. Uruchom ponownie — skrypt załaduje `Runs/Train/weights/last.pt` i kontynuuje od ostatniego checkpointu

```powershell
docker compose -f DockerCompose.yml up
```

Po wznowieniu przywróć `RESUME=false` na przyszłość.

---

## CUDA out of memory

Objaw w logach: `RuntimeError: CUDA out of memory`

**Rozwiązanie:** zmniejsz batch w `DockerCompose.yml`:

```yaml
- BATCH=8    # zamiast 16
```

Dla `imgsz=640` i YOLOv8n:
- `BATCH=16` — ok. 4–5 GB VRAM (powinno działać na 16 GB)
- `BATCH=8`  — ok. 2–3 GB VRAM (bezpieczny fallback)
- `BATCH=32` — ok. 7–8 GB VRAM (jeśli chcesz przyspieszyć)

---

## Przyspieszenie treningu (opcjonalnie)

Jeśli masz dużo wolnego RAM systemowego (≥32 GB), możesz załadować dataset do pamięci.  
W `TrainModel.py` zmień `cache=False` na `cache='ram'` — każda epoka będzie szybsza.

---

## Oczekiwane wyniki po 100 epokach

| Metryka        | Oryginał (5 epok CPU) | Cel (100 epok GPU) |
|----------------|-----------------------|--------------------|
| mAP50          | ~0.16                 | 0.30–0.40          |
| mAP50-95       | ~0.089                | 0.18–0.25          |

VisDrone to trudny zbiór (małe, gęste obiekty z droną) — wyniki rzędu 0.35 mAP50 są dobre.
