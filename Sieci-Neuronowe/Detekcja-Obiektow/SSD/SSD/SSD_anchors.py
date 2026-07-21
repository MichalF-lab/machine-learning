import math
from itertools import product

import torch

SSD_CONFIG = {
    "feature_map_sizes": [38, 19, 10, 5, 3, 1],

    "scales": [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],

    "aspect_ratios": [         # skale: szerokość/wysokość
        [1, 2, 0.5],           # mapa 38×38 (3 anchory + 1 kwadrat)
        [1, 2, 3, 0.5, 0.333], # mapa 19×19 (5 anchory + 1 kwadrat)
        [1, 2, 3, 0.5, 0.333], # mapa 10×10 (5 anchory + 1 kwadrat)
        [1, 2, 3, 0.5, 0.333], # mapa 5×5 (5 anchory + 1 kwadrat)
        [1, 2, 0.5],           # mapa 3×3 (3 anchory + 1 kwadrat)
        [1, 2, 0.5],           # mapa 1×1: (3 anchory + 1 kwadrat)
    ],
}

def generate_anchors(config = SSD_CONFIG, image_size = 300):
    anchors = []

    feature_map_sizes = config["feature_map_sizes"]
    scales = config["scales"]
    aspect_ratios_list = config["aspect_ratios"]

    for k, f_size in enumerate(feature_map_sizes):
        s_k = scales[k]
        s_k_next = scales[k + 1]

        for i, j in product(range(f_size), repeat=2):
            # środek komórki (i, j)
            cx = (j + 0.5) / f_size
            cy = (i + 0.5) / f_size

            for ar in aspect_ratios_list[k]:
                # szerokość i wysokość anchora
                # ar = w/h, więc: w = s_k * sqrt(ar), h = s_k / sqrt(ar)
                w = s_k * math.sqrt(ar)
                h = s_k / math.sqrt(ar)
                anchors.append([cx, cy, w, h])

            # Dodatkowy anchor o skali s_k i s_k_next
            s_prime = math.sqrt(s_k * s_k_next)
            anchors.append([cx, cy, s_prime, s_prime])

    anchors = torch.tensor(anchors, dtype=torch.float32)
    anchors = anchors.clamp(0.0, 1.0)

    return anchors


if __name__ == "__main__":
    anchors = generate_anchors()
    print(f"Łączna liczba anchorów: {anchors.shape[0]}")   # 8732
    print(f"Kształt tensora: {anchors.shape}")             # [8732, 4]
    print(f"Przykładowy anchor: {anchors[6]}")             # środek + rozmiar
    print(f"Min wartość: {anchors.min():.4f}")             # >= 0
    print(f"Max wartość: {anchors.max():.4f}")             # <= 1