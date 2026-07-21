"""
Animacje "samochodzik na wzgórzu" — wizualizacja MountainCar-v0 narysowana
własnoręcznie w matplotlib (bez pygame), na podstawie PRAWDZIWYCH trajektorii
(pozycja/prędkość) nagranych co 25 epizodów podczas treningu DQN + Wariant D
(PBRS) w kontenerze Docker z GPU (zob. train_for_animation.py).

Wejscie: animacje/car_trajectories.json
Wyjscie: animacje/05_car_on_hill_progression.gif, animacje/06_car_before_after.gif
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.transforms import Affine2D
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "animacje")

with open(os.path.join(OUT_DIR, "car_trajectories.json"), encoding="utf-8") as f:
    TRAJ = json.load(f)

GOAL_X = 0.5
TRACK_X = np.linspace(-1.2, 0.6, 400)
HOLD_FRAMES = 8  # ile klatek "postoju" na końcu każdego epizodu, przed przejściem dalej
STRIDE = 2  # co która klatka trajektorii (zmniejsza rozmiar pliku, ruch nadal płynny)


def height(x):
    return np.sin(3 * x)


def slope_angle_deg(x):
    return np.degrees(np.arctan(3 * np.cos(3 * x)))


def draw_track(ax):
    ax.plot(TRACK_X, height(TRACK_X), color="#6b4226", lw=3, zorder=1)
    ax.fill_between(TRACK_X, height(TRACK_X), -1.6, color="#3a2a1a", alpha=0.6, zorder=0)
    ax.plot([GOAL_X, GOAL_X], [height(GOAL_X), height(GOAL_X) + 0.35], color="#e63946", lw=2, zorder=2)
    ax.fill([GOAL_X, GOAL_X + 0.13, GOAL_X], [height(GOAL_X) + 0.35, height(GOAL_X) + 0.30, height(GOAL_X) + 0.25],
            color="#e63946", zorder=2)
    ax.set_xlim(-1.3, 0.75)
    ax.set_ylim(-1.5, 1.4)
    ax.set_facecolor("#0b132b")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_car_artist(ax, color="#f4a261"):
    body = plt.Rectangle((-0.045, -0.02), 0.09, 0.05, color=color, zorder=4)
    ax.add_patch(body)
    wheel1 = plt.Circle((-0.025, -0.025), 0.022, color="#1a1a1a", zorder=5)
    wheel2 = plt.Circle((0.025, -0.025), 0.022, color="#1a1a1a", zorder=5)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)
    return body, wheel1, wheel2


def place_car(ax, parts, x):
    y = height(x)
    angle = slope_angle_deg(x)
    body, wheel1, wheel2 = parts
    t = Affine2D().rotate_deg(angle).translate(x, y + 0.045) + ax.transData
    body.set_transform(t)
    t_w1 = Affine2D().rotate_deg(angle).translate(x - 0.025, y + 0.02) + ax.transData
    t_w2 = Affine2D().rotate_deg(angle).translate(x + 0.025, y + 0.02) + ax.transData
    wheel1.set_transform(t_w1)
    wheel2.set_transform(t_w2)


def progression_animation():
    checkpoints = [k for k in sorted(TRAJ.keys(), key=int)]

    fig, ax = plt.subplots(figsize=(8, 5.3))
    fig.patch.set_facecolor("#0b132b")
    draw_track(ax)
    parts = make_car_artist(ax, color="#f4a261")
    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, color="white", fontsize=15,
                     fontweight="bold", ha="center")

    # Budujemy jedną długą sekwencję klatek: każdy checkpoint -> jego trajektoria + "postój"
    segments = []  # (episode_label, positions)
    for key in checkpoints:
        positions = TRAJ[key]["positions"][::STRIDE]
        frames_for_segment = positions + [positions[-1]] * HOLD_FRAMES
        segments.append((key, frames_for_segment))

    frame_index = []  # (segment_idx, local_frame)
    for seg_idx, (_, frames_for_segment) in enumerate(segments):
        for local_frame in range(len(frames_for_segment)):
            frame_index.append((seg_idx, local_frame))

    def update(i):
        seg_idx, local_frame = frame_index[i]
        key, frames_for_segment = segments[seg_idx]
        x = frames_for_segment[local_frame]
        place_car(ax, parts, x)
        title.set_text(f"Epizod {key} / 500")
        return list(parts) + [title]

    anim = FuncAnimation(fig, update, frames=len(frame_index), interval=35, blit=False)
    out_path = os.path.join(OUT_DIR, "05_car_on_hill_progression.gif")
    anim.save(out_path, writer=PillowWriter(fps=20), dpi=85)
    plt.close(fig)
    print("Zapisano:", out_path)


def before_after_animation():
    left_key, right_key = "0", "250"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor("#0b132b")

    draw_track(axL)
    draw_track(axR)
    axL.set_title(f"Epizod {left_key}", color="#9a9a9a", fontsize=12, pad=10)
    axR.set_title(f"Epizod {right_key}", color="#06d6a0", fontsize=12, pad=10)

    partsL = make_car_artist(axL, color="#9a9a9a")
    partsR = make_car_artist(axR, color="#06d6a0")

    posL = TRAJ[left_key]["positions"]
    posR = TRAJ[right_key]["positions"]

    fig.suptitle("MountainCar-v0 — ten sam start, inna polityka", color="white", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    max_len = max(len(posL), len(posR))

    def update(frame):
        idxL = min(frame, len(posL) - 1)
        idxR = min(frame, len(posR) - 1)
        place_car(axL, partsL, posL[idxL])
        place_car(axR, partsR, posR[idxR])
        return list(partsL) + list(partsR)

    anim = FuncAnimation(fig, update, frames=max_len, interval=45, blit=False)
    out_path = os.path.join(OUT_DIR, "06_car_before_after.gif")
    anim.save(out_path, writer=PillowWriter(fps=22))
    plt.close(fig)
    print("Zapisano:", out_path)


if __name__ == "__main__":
    progression_animation()
    before_after_animation()
    print("Gotowe.")
