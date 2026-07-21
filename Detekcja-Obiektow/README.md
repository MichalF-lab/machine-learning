# Detekcja obiektów — YOLO, SSD, Faster R-CNN

*Projekt grupowy (Grupa 2): trzy architektury detekcji obiektów implementowane od podstaw, potem dotrenowane w Dockerze na GPU na BCCD i VisDrone.*

## 📖 Opis

Implementacja i porównanie trzech podejść do detekcji obiektów:
- **YOLO** — model jednoetapowy (grid + anchor boxes), własna implementacja straty i ewaluacji
- **SSD** — Single Shot Detector, własne anchory i model
- **Faster R-CNN** — dwuetapowy detektor (RPN + generowanie propozycji + IoU/offsety), od anchorów po pełną pętlę treningową

Po wstępnej wersji grupowej model Faster R-CNN i pipeline YOLO zostały dotrenowane osobno w Dockerze z GPU na dwóch datasetach (BCCD — komórki krwi, VisDrone — obiekty z drona), z analizą błędów na koniec.

## 📂 Struktura

| Folder | Zawartość |
|---|---|
| `YOLO/yolo/` | Model, funkcja straty i ewaluacja YOLO od podstaw + wykresy treningowe |
| `SSD/SSD/` | Model i anchory SSD + wstęp teoretyczny (PDF) |
| `Faster_R_CNN/Faster_R_CNN/` | Faster R-CNN od podstaw: anchory, propozycje, IoU, obliczanie strat i offsetów |
| `drive_content/` | Wspólne pliki zespołu (loader danych, notatnik `Projekt2.ipynb`, wstęp do SSD) |
| `loaders_content/loaders/` | Parsery datasetów (BCCD, BDD100K, VisDrone) + pobieranie danych |
| `FrcnnTrain/` | Dotrenowana wersja Faster R-CNN w Dockerze/GPU — osobne przebiegi na BCCD i VisDrone (`Runs/bccd`, `Runs/visdrone` — metryki, bez wag modelu) |
| `YoloBccdTrain/` | YOLOv8n (Ultralytics) trenowany na BCCD w Dockerze/GPU — patrz [ReadMe](./YoloBccdTrain/ReadMe.md) |
| `YoloVisDroneTrain/` | YOLOv8n trenowany na VisDrone2019-DET w Dockerze/GPU — patrz [ReadMe](./YoloVisDroneTrain/ReadMe.md) |
| `error_analysis/` | Analiza błędów detekcji na BCCD, BDD100K i VisDrone (`output/*.png`) |
| `GRUPA2_Detekcja_Obiektow_Postepy.pptx` | Prezentacja postępów grupy |

## ⚠️ Uwagi

Surowe datasety (BCCD, VisDrone) oraz wagi modeli z pośrednich epok są pomijane w repozytorium (`.gitignore`) — datasety pobierają się automatycznie skryptami setupowymi (`App/BccdSetup.py` i analogiczne), a checkpointy treningowe łącznie ważyły >2,5 GB. Zostają wyniki: metryki, wykresy (PR/F1/confusion matrix), przykładowe predykcje i finalne, niewielkie modele tam, gdzie to nie kolidowało z rozmiarem repo.

## 🛠️ Technologie

| Technologia | Szczegóły |
|---|---|
| Python | PyTorch, Ultralytics YOLOv8, OpenCV |
| Infrastruktura | Docker + GPU (CUDA), datasety BCCD / VisDrone2019-DET / BDD100K |

## 👤 Autor

Michał Frąckowiak (część indywidualna: `FrcnnTrain`, `YoloBccdTrain`, `YoloVisDroneTrain`, `error_analysis`) + Grupa 2
