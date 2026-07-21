import pandas as pd
import os

def save_logs(episode_logs, save_dir="results/logs", filename="log.csv"):
    # Zapis CSV z listy słowników

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    pd.DataFrame(episode_logs).to_csv(path, index=False)
    print(f"[Logger] Zapisano {len(episode_logs)} epizodów → {path}")
    return path
