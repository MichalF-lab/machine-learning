"""
Animacje pokazujace jak agenci RL (DQN, REINFORCE) uczą się rozwiązywać
MountainCar-v0, na podstawie prawdziwych logów z treningu
(results/logs/*.csv) — baseline + warianty reward-shaping A-E.

Wyjscie: D:\\CLCO\\Projekt_3\\animacje\\*.gif
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(BASE, "results", "logs")
OUT_DIR = os.path.join(BASE, "animacje")
os.makedirs(OUT_DIR, exist_ok=True)

VARIANT_DESC = {
    "baseline": "Baseline (brak shaping)",
    "variant_A": "Wariant A (bonus za pozycję)",
    "variant_B": "Wariant B (bonus za prędkość)",
    "variant_C": "Wariant C (hybryda A+B)",
    "variant_D": "Wariant D (PBRS — potencjał)",
    "variant_E": "Wariant E (bonus za rekord wysokości)",
}

COLORS = {
    "baseline": "#9a9a9a",
    "reinforce_baseline": "#48cae4",
    "variant_A": "#f4a261",
    "variant_B": "#e76f51",
    "variant_C": "#2a9d8f",
    "variant_D": "#264653",
    "variant_E": "#8338ec",
}


def load(name):
    return pd.read_csv(os.path.join(LOGS, name))


def race_animation(series_specs, title, out_name, n_episodes=500, every=2):
    """series_specs: list of (label, key, dataframe)"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.patch.set_facecolor("#0b132b")
    ax.set_facecolor("#0b132b")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#3a3a5a")
    ax.grid(alpha=0.2, color="white")
    ax.set_xlim(1, n_episodes)
    all_vals = np.concatenate([df["avg_100"].values for _, _, df in series_specs])
    ax.set_ylim(all_vals.min() - 10, all_vals.max() + 10)
    ax.axhline(-110, color="#06d6a0", ls="--", lw=1, alpha=0.6)
    ax.text(n_episodes * 0.99, -110, " próg „rozwiązane” (orientacyjny)", color="#06d6a0",
            fontsize=7, ha="right", va="bottom")
    ax.set_xlabel("Epizod", color="white")
    ax.set_ylabel("Średnia nagroda (ostatnie 100 epizodów)", color="white")
    ax.set_title(title, color="white", fontsize=13, pad=12)

    lines, dots, texts = {}, {}, {}
    for label, key, df in series_specs:
        color = COLORS.get(key, "white")
        (line,) = ax.plot([], [], color=color, lw=2.2, label=label)
        (dot,) = ax.plot([], [], "o", color=color, ms=6)
        txt = ax.text(0, 0, "", color=color, fontsize=8, fontweight="bold", va="center")
        lines[key] = line
        dots[key] = dot
        texts[key] = txt

    ax.legend(loc="upper left", facecolor="#0b132b", edgecolor="#3a3a5a", fontsize=8, labelcolor="white")
    epoch_label = ax.text(0.02, 0.04, "", transform=ax.transAxes, color="white", fontsize=11, fontweight="bold")

    frames = list(range(1, n_episodes + 1, every))

    def update(frame_idx):
        ep = frames[frame_idx]
        for label, key, df in series_specs:
            x = df["episode"].values[:ep]
            y = df["avg_100"].values[:ep]
            lines[key].set_data(x, y)
            dots[key].set_data([x[-1]], [y[-1]])
            texts[key].set_position((x[-1] + n_episodes * 0.012, y[-1]))
            texts[key].set_text(f"{y[-1]:.0f}")
        epoch_label.set_text(f"Epizod {ep}/{n_episodes}")
        return list(lines.values()) + list(dots.values()) + list(texts.values()) + [epoch_label]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    out_path = os.path.join(OUT_DIR, out_name)
    anim.save(out_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print("Zapisano:", out_path)


def noise_vs_trend_animation(df, title, out_name, n_episodes=500, every=2):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.patch.set_facecolor("#0b132b")
    ax.set_facecolor("#0b132b")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#3a3a5a")
    ax.grid(alpha=0.2, color="white")
    ax.set_xlim(1, n_episodes)
    ax.set_ylim(df["reward"].min() - 10, df["reward"].max() + 10)
    ax.set_xlabel("Epizod", color="white")
    ax.set_ylabel("Nagroda", color="white")
    ax.set_title(title, color="white", fontsize=13, pad=12)

    (raw_line,) = ax.plot([], [], color="#f4a261", lw=0.7, alpha=0.5, label="Nagroda epizodu (surowa)")
    (smooth_line,) = ax.plot([], [], color="#264653", lw=2.4, label="Średnia (100 epizodów)")
    (dot,) = ax.plot([], [], "o", color="#264653", ms=6)
    ax.legend(loc="upper left", facecolor="#0b132b", edgecolor="#3a3a5a", fontsize=8, labelcolor="white")
    epoch_label = ax.text(0.02, 0.92, "", transform=ax.transAxes, color="white", fontsize=11, fontweight="bold")

    frames = list(range(1, n_episodes + 1, every))

    def update(frame_idx):
        ep = frames[frame_idx]
        x = df["episode"].values[:ep]
        raw_line.set_data(x, df["reward"].values[:ep])
        smooth_line.set_data(x, df["avg_100"].values[:ep])
        dot.set_data([x[-1]], [df["avg_100"].values[ep - 1]])
        epoch_label.set_text(f"Epizod {ep}/{n_episodes}")
        return [raw_line, smooth_line, dot, epoch_label]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    out_path = os.path.join(OUT_DIR, out_name)
    anim.save(out_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print("Zapisano:", out_path)


if __name__ == "__main__":
    dqn_base = load("DQN_MountainCar-v0.csv")
    rein_base = load("REINFORCE_MountainCar-v0.csv")

    race_animation(
        [("DQN (baseline)", "baseline", dqn_base), ("REINFORCE (baseline)", "reinforce_baseline", rein_base)],
        "MountainCar-v0 — DQN vs REINFORCE (brak reward shaping)\nProblem rzadkiej nagrody: oba modele zostają przy -200",
        "01_dqn_vs_reinforce_baseline.gif",
    )

    dqn_variants = [("DQN " + VARIANT_DESC["baseline"], "baseline", dqn_base)]
    for v in ["variant_A", "variant_B", "variant_C", "variant_D", "variant_E"]:
        dqn_variants.append(("DQN " + VARIANT_DESC[v], v, load(f"DQN_MountainCar-v0_{v}.csv")))
    race_animation(
        dqn_variants,
        "DQN — wpływ reward shaping na tempo uczenia (MountainCar-v0)",
        "02_dqn_reward_shaping_race.gif",
    )

    rein_variants = [("REINFORCE " + VARIANT_DESC["baseline"], "baseline", rein_base)]
    for v in ["variant_A", "variant_B", "variant_C", "variant_D", "variant_E"]:
        rein_variants.append(("REINFORCE " + VARIANT_DESC[v], v, load(f"REINFORCE_MountainCar-v0_{v}.csv")))
    race_animation(
        rein_variants,
        "REINFORCE — wpływ reward shaping na tempo uczenia (MountainCar-v0)",
        "03_reinforce_reward_shaping_race.gif",
    )

    noise_vs_trend_animation(
        load("DQN_MountainCar-v0_variant_D.csv"),
        "DQN + Wariant D (PBRS) — surowa nagroda vs trend uczenia",
        "04_dqn_variant_D_noise_vs_trend.gif",
    )

    print("Gotowe. Wszystkie animacje w:", OUT_DIR)
