# Uczenie Maszynowe

*Eksperymenty i implementacje z uczenia maszynowego — od podstaw i z użyciem PyTorch/GPU.*

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

## 📖 Opis

Zbiór eksperymentów z uczenia maszynowego i głębokiego: sieci neuronowe implementowane od podstaw, wersje z akceleracją GPU, Monte Carlo Tree Search oraz klasyczne modele regresji.

## 📂 Struktura

| Folder | Plik | Zawartość |
|---|---|---|
| `sieci-od-zera/` | `NNfromNeuron.py`, `NNfromLayer.py` | Sieć neuronowa od podstaw (pojedynczy neuron → warstwa) |
| | `NNonGPU.py` | Sieć neuronowa z obliczeniami na GPU (CuPy) |
| | `siec_neuronowa_mnist.py` | Ręczna implementacja sieci (klasy `neuron`, `siec`) trenowana na MNIST |
| | `neuron_szkic.py` | Szkic implementacji pojedynczego neuronu |
| `pytorch/` | `cifar10_siec_konwolucyjna_podstawowa.py`, `convolutionnetwork.py` | Konwolucyjne sieci neuronowe (PyTorch) na zbiorze CIFAR-10 |
| | `siec_mnist_regresja_grzbietowa.py`, `deep_learning_pytorch.py` | Sieci neuronowe w PyTorch trenowane na MNIST (pierwsza z warstwą wyjściową dopasowaną regresją grzbietową) |
| | `ucf101_cnn_lstm.py` | Model CNN-LSTM do klasyfikacji wideo na zbiorze UCF-101 |
| `regresja/` | `LinearRegr.py`, `RidgeRegr.py` | Własne implementacje regresji liniowej i grzbietowej |
| | `test_regresja_liniowa.py`, `test_regresja_grzbietowa.py` | Testy porównujące powyższe implementacje z `scikit-learn` |
| `dane/` | `mnist.pkl.gz` | Zbiór danych MNIST |
| | `cifar-10-batches-py/` | Zbiór danych CIFAR-10 |
| *(root)* | `ConnectFourMCST.py` | Monte Carlo Tree Search dla gry Connect Four |

## 🛠️ Technologie

| Technologia | Szczegóły |
|---|---|
| Python | biblioteki: PyTorch (`torch`, `torchvision`), scikit-learn, OpenCV (`cv2`), CuPy (obliczenia na GPU) |

## ⚠️ Uwagi

Skrypty w `pytorch/` ładujące CIFAR-10 (`root='./data'`) i MNIST zakładają dane w folderze `data/` względem miejsca uruchomienia — po przeniesieniu danych do `dane/` należy uruchamiać je z odpowiednim `cwd` albo poprawić ścieżkę; `torchvision.datasets.CIFAR10` pobierze dane automatycznie, jeśli nie znajdzie ich lokalnie.

## 👤 Autor

Michał Frąckowiak
