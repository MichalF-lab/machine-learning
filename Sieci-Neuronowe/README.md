# Sieci Neuronowe

*Trzy projekty semestralne z przedmiotu Sieci Neuronowe — detekcja obiektów, porównanie architektur ViT/CNN i optymalizacja agenta RL.*

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

## 📖 Opis

Repozytorium zbiera wszystkie projekty zaliczeniowe z przedmiotu Sieci Neuronowe — dwa projekty grupowe i jeden indywidualny, każdy jako osobny podfolder z własnym README.

## 📂 Struktura

| Folder | Zawartość |
|---|---|
| [`Detekcja-Obiektow/`](./Detekcja-Obiektow) | Detekcja obiektów: YOLO, SSD i Faster R-CNN od podstaw (projekt grupowy) + dopracowane wersje treningowe w Dockerze/GPU na BCCD i VisDrone |
| [`ViT-vs-CNN/`](./ViT-vs-CNN) | Analiza porównawcza Vision Transformer vs CNN na CIFAR-10 i ImageNet (projekt grupowy) |
| [`RL-Gymnasium/`](./RL-Gymnasium) | Optymalizacja agenta RL (DQN/REINFORCE/PPO) w środowisku MountainCar-v0 z reward shaping (projekt grupowy) |

## 🛠️ Technologie

| Technologia | Szczegóły |
|---|---|
| Python | PyTorch (`torch`, `torchvision`), Ultralytics YOLOv8, Gymnasium, OpenCV, scikit-learn |
| Infrastruktura | Docker + GPU (CUDA) do treningu modeli |

## 👤 Autor

Michał Frąckowiak
