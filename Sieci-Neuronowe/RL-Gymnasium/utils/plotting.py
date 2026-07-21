import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_rewards(results, env_name, save_dir="plots"):
    # Rysuje krzywe uczenia dla agentów

    os.makedirs(save_dir, exist_ok=True)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#FF00FF"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, rewards) in enumerate(results.items()):
        episodes = np.arange(1, len(rewards) + 1)
        rolling = pd.Series(rewards).rolling(window=100, min_periods=1).mean().values
        color = colors[i % len(colors)]

        ax.plot(episodes, rewards, alpha=0.25, color=color, linewidth=0.8)
        ax.plot(episodes, rolling, color=color, linewidth=2, label=name)

    ax.set_title(f"Krzywe uczenia – {env_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epizod")
    ax.set_ylabel("Łączna nagroda")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    agents = "_".join(results.keys())
    path = os.path.join(save_dir, f"{agents}_{env_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Zapisano → {path}")
    return path
