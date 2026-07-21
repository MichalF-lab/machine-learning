#   Uruchomienie:
#      python run_baseline.py --agent dqn
#      python run_baseline.py --agent reinforce
#      python run_baseline.py --agent all
#      itd.


import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
from agents.dqn_agent       import DQNAgent
from agents.reinforce_agent import REINFORCEAgent
from agents.ppo_agent       import PPOAgent
from utils.training         import train, evaluate
from utils.logger           import save_logs
from utils.plotting         import plot_rewards


ENV_NAME = "MountainCar-v0"

DQN_CONFIG = {
    "n_episodes":         500,
    "max_steps":          1000,
    "lr":                 1e-3,
    "gamma":              0.99,
    "epsilon_start":      1.0,
    "epsilon_min":        0.01,
    "epsilon_decay":      0.995,  # trzeba sprawdzić
    "batch_size":         64,
    "buffer_size":        100_000,
    "target_update_freq": 10,
    "hidden_dims":        [128, 128], # może być 64 czy 256
}

REINFORCE_CONFIG = {
    "n_episodes":  500,
    "max_steps":   1000,
    "lr":          1e-3,
    "gamma":       0.99,
    "hidden_dims": [128, 128],
}

PPO_CONFIG = {
    "n_episodes": 500,
    "max_steps": 1000,
    "lr": 3e-4,
    "gamma": 0.99,
    "eps_clip": 0.2,
    "K_epochs": 4,
    "hidden_dims": [128, 128],
}

# Pomocnicze
def get_env_dims(env_name):
    env = gym.make(env_name)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    return state_dim, action_dim


def run_agent(agent_name):
    state_dim, action_dim = get_env_dims(ENV_NAME)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    if agent_name == "DQN":
        agent  = DQNAgent(state_dim, action_dim, DQN_CONFIG)
        config = DQN_CONFIG
    elif agent_name == "REINFORCE":
        agent  = REINFORCEAgent(state_dim, action_dim, REINFORCE_CONFIG)
        config = REINFORCE_CONFIG
    elif agent_name == "PPO":
        agent = PPOAgent(state_dim, action_dim, PPO_CONFIG)
        config = PPO_CONFIG
    else:
        raise ValueError(f"Nieznany agent: {agent_name}")

    print(f"\n[{agent_name}] Środowisko: {ENV_NAME} | Stan: {state_dim}D | Akcje: {action_dim}")

    # Trening
    logs = train(agent, ENV_NAME, config)

    # Zapis CSV
    save_logs(logs, save_dir="results/logs", filename=f"{agent_name}_{ENV_NAME}.csv")

    # Zapis modelu
    agent.save(f"results/models/{agent_name}_{ENV_NAME}.pt")

    # Ewaluacja
    print(f"\n--- Ewaluacja {agent_name} ---")
    eval_rewards = evaluate(agent, ENV_NAME, n_episodes=20)

    # Wykres
    rewards = [row["reward"] for row in logs]
    plot_rewards({agent_name: rewards}, ENV_NAME, save_dir="plots")

    return rewards, eval_rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "reinforce", "ppo", "all"])
    args = parser.parse_args()   # python run_baseline.py --agent dqn

    all_rewards = {}

    if args.agent in ("dqn", "all"):
        rewards, _ = run_agent("DQN")
        all_rewards["DQN"] = rewards

    if args.agent in ("reinforce", "all"):
        rewards, _ = run_agent("REINFORCE")
        all_rewards["REINFORCE"] = rewards

    if args.agent in ("ppo", "all"):
        rewards, _ = run_agent("PPO")
        all_rewards["PPO"] = rewards


    if args.agent == "all":
        plot_rewards(all_rewards, ENV_NAME, save_dir="plots")

    print("\n Gotowe. Wyniki są w 'results/', wykresy są w 'plots/'")

if __name__ == "__main__":
    main()
