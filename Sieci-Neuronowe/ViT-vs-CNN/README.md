# GRUPA 3 - Vision Transformer vs CNN
## Analiza porównawcza architektur na CIFAR-10 i ImageNet

---

## 📊 Zawartość Projektu

### Notebooki Jupyter
1. **Grupa3_ViT_vs_CNN_Analiza.ipynb** - Główna analiza
   - Porównanie CNN vs ViT na CIFAR-10 (6 eksperymentów)
   - Porównanie CNN vs ViT na ImageNet (6 eksperymentów)
   - Wykresy: dokładność, czas treningu, liczba parametrów
   - Krzywe treningu dla każdego eksperymentu
   - Porównanie zbiorów danych
   - **Główne odkrycie:** ViT_Best osiąga 72.73% na CIFAR-10, CNN osiąga 72.04%

2. **Bonus_150epochs_Analiza.ipynb** - Analiza rozszerzona
   - ViT z 150 epokami na CIFAR-10
   - **Wynik:** 81.0% dokładności!
   - Porównanie: 10 epok vs 150 epok
   - Milestones - postęp treningu
   - **Wniosek:** Długi trening wart dodatkowego czasu

### Dane JSON
- `porownanie_wynikow.json` - Główne wyniki (CIFAR-10 + ImageNet)
- `long_training_vit_best_150ep.json` - Wyniki 150 epok
- `bonus.json`, `bonus2.json` - Dodatkowe eksperymenty
- `imagenet_112x112.json` - Dodatkowe wyniki ImageNet

---

## 🚀 Uruchomienie

### Opcja 1: Na dysku (lokalnie)

#### Wymagania
```bash
pip install jupyter matplotlib seaborn pandas numpy
```

#### Uruchomienie
```bash
# Wejdź do katalogu projektu
cd ViT-vs-CNN

# Uruchom Jupyter
jupyter notebook

# Otwórz w przeglądarce:
# - Grupa3_ViT_vs_CNN_Analiza.ipynb
# - Bonus_150epochs_Analiza.ipynb
```

### Opcja 2: W Dockerze

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    jupyter \
    matplotlib \
    seaborn \
    pandas \
    numpy

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
```

#### Uruchomienie w Dockerze
```bash
# Zbuduj obraz
docker build -t vit-vs-cnn:latest .

# Uruchom kontener
docker run -p 8888:8888 -v $(pwd):/app vit-vs-cnn:latest

# Jupyter będzie dostępny na: http://localhost:8888
```

#### Dla Windows (PowerShell)
```powershell
docker build -t vit-vs-cnn:latest .
docker run -p 8888:8888 -v ${pwd}:/app vit-vs-cnn:latest
```

---

## 📈 Główne Wyniki

### CIFAR-10
| Model | Test Accuracy | Training Time | Parameters |
|-------|--------------|---------------|-----------|
| CNN (10 ep) | **72.04%** | 106s | 4.3M |
| ViT_Best (10 ep) | 72.73% | 1240s | 4.7M |
| ViT_Best (150 ep) | **81.0%** | 18424s | 4.7M |

### ImageNet
| Model | Test Accuracy | Training Time | Parameters |
|-------|--------------|---------------|-----------|
| CNN (10 ep) | 62.37% | 292s | 4.3M |
| ViT_Best (10 ep) | **66.52%** | 295s | 4.7M |

---

## 🎯 Kluczowe Wnioski

### 1. Na CIFAR-10:
- **CNN** - szybki, wysoką dokładność (72%), ideał dla szybkich eksperymentów
- **ViT** - lepszy przy długim treningu (81% po 150 epokach!)
- Transformery skalują się znacznie lepiej z ilością epok

### 2. Na ImageNet:
- **ViT** wyraźnie lepszy (66.52% vs 62.37% dla CNN)
- Transformery lepiej generalizują na większych/złożonych datasetach
- CNN ma problemy z transferem do bardziej złożonych danych

### 3. Efektywność:
- CNN lepszy `czas/dokładność` dla krótkich treningów
- ViT wart inwestycji czasu dla finalnych modeli
- ViT_Best + 150 epok = najlepszy wynik (81%)

### 4. Architektura:
- ViT_Small (0.4M param) - zbyt mały
- ViT_Medium (4.7M param) - optymalny
- ViT_Large (7.1M param) - overkill na CIFAR-10

---

## 📁 Struktura Plików

```
ViT-vs-CNN/
├── Grupa3_ViT_vs_CNN_Analiza.ipynb          # Główna analiza
├── Bonus_150epochs_Analiza.ipynb             # Analiza bonusowa
├── porownanie_wynikow.json                   # Dane główne
├── long_training_vit_best_150ep.json         # Dane 150 epok
├── bonus.json, bonus2.json                   # Dodatkowe dane
├── imagenet_112x112.json                     # Dane ImageNet
├── wykresy/                                  # Wykresy wynikowe (accuracy, loss, porównania)
├── notebooki-zespolu/                        # Indywidualne notatniki członków zespołu
├── prezentacja/                              # Finalne notatniki + prezentacja (pptx)
├── animacje/                                 # Animowane wykresy porównawcze (gif)
├── Dockerfile                                # Konfiguracja Docker
└── README.md                                 # Ta dokumentacja
```

---

## 💾 Jak Wczytać Dane w Notatniku

```python
import json

# Główne dane
with open('porownanie_wynikow.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Wyniki 150 epok
with open('long_training_vit_best_150ep.json', 'r', encoding='utf-8') as f:
    long_data = json.load(f)

# Dostęp do danych
cifar10_results = data['cifar10']
imagenet_results = data['imagenet']
long_train_cifar = long_data['cifar10']
```

---

## 🔧 Zainstalowane Pakiety

Notebooki wymagają:
- `jupyter` - interaktywne notebooki
- `matplotlib` - wykresy
- `seaborn` - stylizacja wykresów
- `pandas` - manipulacja danymi
- `numpy` - obliczenia numeryczne

---

## 📝 Opis Eksperymentów

### CNN (Standard)
- 5 i 10 epok
- ~4.3M parametrów
- Konwolucyjne warstwy
- Bazowa linia porównawcza

### ViT (Vision Transformer)
- **Small**: 0.4M parametrów, słaby na małych datasetach
- **Medium**: 4.7M parametrów, optymalny
- **Large**: 7.1M parametrów, overkill
- **Best**: Tuned config, najlepszy wynik
- **Long Training**: 150 epok, maksymalny wynik

---

## 🎓 Nauki Wyniesione

1. **Transformery vs CNN**: Nowoczesne podejście (transformery) wynika lepiej na złożonych zadaniach
2. **Skalowanie**: Vision Transformers skalują się lepiej z ilością danych i epok
3. **Trade-off**: CNN - szybko, ViT - dokładnie (po investycji czasu)
4. **Early Stopping**: ViT nie osiąga plateau do 150 epok na CIFAR-10

---

## 📞 Kontakt / Notatki

- **Data**: 2026-04-14
- **GPU**: RTX 5060 Ti 16GB
- **Framework**: PyTorch
- **Dataset**: CIFAR-10 (60k samples), ImageNet (1.3M samples)
