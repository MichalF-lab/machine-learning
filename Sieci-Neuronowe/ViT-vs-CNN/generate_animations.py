"""
Generuje animacje (GIF) pokazujace jak model "sie uczy" na podstawie
zapisanych historii treningu w porownanie_wynikow.json oraz
long_training_vit_best_150ep.json.

Wyjscie: D:\\CLCO\\Projekt_1\\animacje\\*.gif
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "animacje")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(BASE, "porownanie_wynikow.json"), encoding="utf-8") as f:
    CMP = json.load(f)

with open(os.path.join(BASE, "long_training_vit_best_150ep.json"), encoding="utf-8") as f:
    LONG = json.load(f)

COLORS = {
    "CNN_5": "#e07a5f",
    "CNN_10": "#bb3e03",
    "ViT_Small": "#90e0ef",
    "ViT_Medium": "#48cae4",
    "ViT_Large": "#0096c7",
    "ViT_Best": "#023e8a",
}


def label_for(entry):
    if entry["model"] == "CNN":
        key = f"CNN_{entry['param_value']}"
        return key, f"CNN ({entry['param_value']} ep)"
    key = entry["experiment"]
    return key, f"{entry['experiment']} ({entry['param_value']} ep)"


def race_animation(dataset_key, title, out_name, metric="test_acc", ylabel="Test accuracy (%)"):
    entries = CMP[dataset_key]
    series = []
    max_epoch = 0
    for e in entries:
        key, label = label_for(e)
        epochs = [h["epoch"] for h in e["history"]]
        values = [h[metric] for h in e["history"]]
        max_epoch = max(max_epoch, max(epochs))
        series.append({"key": key, "label": label, "epochs": epochs, "values": values})

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#0b132b")
    ax.set_facecolor("#0b132b")
    ax.set_xlim(0.5, max_epoch + 0.5)
    pad = 5 if "acc" in metric else 0.1
    all_vals = [v for s in series for v in s["values"]]
    ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)
    ax.set_xlabel("Epoka", color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#3a3a5a")
    ax.grid(alpha=0.2, color="white")

    lines = {}
    dots = {}
    texts = {}
    for s in series:
        color = COLORS.get(s["key"], "white")
        (line,) = ax.plot([], [], color=color, lw=2.2, label=s["label"])
        (dot,) = ax.plot([], [], "o", color=color, ms=6)
        txt = ax.text(0, 0, "", color=color, fontsize=8, fontweight="bold", va="center")
        lines[s["key"]] = line
        dots[s["key"]] = dot
        texts[s["key"]] = txt

    legend = ax.legend(loc="lower right", facecolor="#0b132b", edgecolor="#3a3a5a", fontsize=8, labelcolor="white")

    epoch_label = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white", fontsize=11, fontweight="bold")

    n_frames = max_epoch
    interp = {}
    for s in series:
        xs = np.array(s["epochs"], dtype=float)
        ys = np.array(s["values"], dtype=float)
        full_x = np.arange(1, max_epoch + 1)
        full_y = np.interp(full_x, xs, ys, left=ys[0], right=ys[-1])
        full_y[full_x > xs[-1]] = ys[-1]
        interp[s["key"]] = (full_x, full_y)

    def update(frame):
        ep = frame + 1
        for s in series:
            full_x, full_y = interp[s["key"]]
            idx = min(ep, len(full_x))
            lines[s["key"]].set_data(full_x[:idx], full_y[:idx])
            dots[s["key"]].set_data([full_x[idx - 1]], [full_y[idx - 1]])
            texts[s["key"]].set_position((full_x[idx - 1] + max_epoch * 0.012, full_y[idx - 1]))
            texts[s["key"]].set_text(f"{full_y[idx - 1]:.1f}")
        epoch_label.set_text(f"Epoka {ep}/{max_epoch}")
        return list(lines.values()) + list(dots.values()) + list(texts.values()) + [epoch_label]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=180, blit=False)
    out_path = os.path.join(OUT_DIR, out_name)
    anim.save(out_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("Zapisano:", out_path)


def long_training_animation():
    cifar = LONG["cifar10"]
    history = cifar["history"]
    epochs = np.array([h["epoch"] for h in history], dtype=float)
    train_acc = np.array([h["train_acc"] for h in history], dtype=float)
    test_acc = np.array([h["test_acc"] for h in history], dtype=float)
    loss = np.array([h["loss"] for h in history], dtype=float)
    best_epoch = cifar["best_epoch"]
    best_acc = cifar["best_test_acc"]

    full_x = np.arange(epochs.min(), epochs.max() + 1)
    full_train = np.interp(full_x, epochs, train_acc)
    full_test = np.interp(full_x, epochs, test_acc)
    full_loss = np.interp(full_x, epochs, loss)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("#0b132b")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#3a3a5a")
        ax.grid(alpha=0.2, color="white")
    fig.patch.set_facecolor("#0b132b")

    ax1.set_xlim(full_x.min(), full_x.max())
    ax1.set_ylim(min(full_test.min(), full_train.min()) - 3, max(full_test.max(), full_train.max()) + 3)
    ax1.set_ylabel("Accuracy (%)", color="white")
    ax1.set_title("ViT_Best — 150 epok na CIFAR-10 (jak model sie uczy)", color="white", fontsize=13, pad=10)

    ax2.set_xlim(full_x.min(), full_x.max())
    ax2.set_ylim(full_loss.min() - 0.05, full_loss.max() + 0.05)
    ax2.set_xlabel("Epoka", color="white")
    ax2.set_ylabel("Loss", color="white")

    (l_train,) = ax1.plot([], [], color="#90e0ef", lw=2, label="Train accuracy")
    (l_test,) = ax1.plot([], [], color="#f4a261", lw=2.4, label="Test accuracy")
    (d_train,) = ax1.plot([], [], "o", color="#90e0ef", ms=5)
    (d_test,) = ax1.plot([], [], "o", color="#f4a261", ms=5)
    ax1.axvline(best_epoch, color="#e63946", ls="--", lw=1, alpha=0.7)
    ax1.text(best_epoch, ax1.get_ylim()[1] - 1, f" najlepszy epoch={best_epoch} ({best_acc:.2f}%)",
              color="#e63946", fontsize=8, va="top")
    ax1.legend(loc="lower right", facecolor="#0b132b", edgecolor="#3a3a5a", fontsize=9, labelcolor="white")

    (l_loss,) = ax2.plot([], [], color="#e63946", lw=2, label="Loss")
    (d_loss,) = ax2.plot([], [], "o", color="#e63946", ms=5)
    ax2.legend(loc="upper right", facecolor="#0b132b", edgecolor="#3a3a5a", fontsize=9, labelcolor="white")

    epoch_label = ax1.text(0.02, 0.95, "", transform=ax1.transAxes, color="white", fontsize=11, fontweight="bold")

    n_frames = len(full_x)

    def update(frame):
        idx = frame + 1
        l_train.set_data(full_x[:idx], full_train[:idx])
        l_test.set_data(full_x[:idx], full_test[:idx])
        d_train.set_data([full_x[idx - 1]], [full_train[idx - 1]])
        d_test.set_data([full_x[idx - 1]], [full_test[idx - 1]])
        l_loss.set_data(full_x[:idx], full_loss[:idx])
        d_loss.set_data([full_x[idx - 1]], [full_loss[idx - 1]])
        epoch_label.set_text(f"Epoka {int(full_x[idx - 1])}/150")
        return [l_train, l_test, d_train, d_test, l_loss, d_loss, epoch_label]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=120, blit=False)
    out_path = os.path.join(OUT_DIR, "04_long_training_150ep.gif")
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    print("Zapisano:", out_path)


if __name__ == "__main__":
    race_animation("cifar10", "CIFAR-10 — test accuracy w trakcie treningu", "01_cifar10_test_acc_race.gif")
    race_animation("cifar10", "CIFAR-10 — loss w trakcie treningu", "02_cifar10_loss_curves.gif",
                    metric="loss", ylabel="Loss")
    race_animation("imagenet", "ImageNet (Imagenette) — test accuracy w trakcie treningu",
                    "03_imagenet_test_acc_race.gif")
    long_training_animation()
    print("Gotowe. Wszystkie animacje w:", OUT_DIR)
