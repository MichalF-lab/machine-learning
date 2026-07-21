import sys
import os
import argparse
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))

from agents.dqn_agent       import DQNAgent
from agents.reinforce_agent import REINFORCEAgent
from agents.ppo_agent       import PPOAgent
from utils.training         import train, evaluate
from utils.logger           import save_logs
from utils.plotting         import plot_rewards
from reward_wrappers        import MountainCarRewardWrapper

ENV_NAME = "MountainCar-v0"
CONFIGS = {
    "DQN": {
        "n_episodes": 500, "max_steps": 1000, "lr": 1e-3, "gamma": 0.99,
        "epsilon_start": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.995,
        "batch_size": 64, "buffer_size": 100_000, "target_update_freq": 10, "hidden_dims": [128, 128],
    },
    "REINFORCE": {
        "n_episodes": 500, "max_steps": 1000, "lr": 1e-3, "gamma": 0.99, "hidden_dims": [128, 128],
    },
    "PPO": {
    "n_episodes": 500, "max_steps": 1000, "lr": 3e-4, "gamma": 0.99, "eps_clip": 0.2, "K_epochs": 4, "hidden_dims": [128, 128],
    }
}

def train_with_shaping(agent: DQNAgent|REINFORCEAgent|PPOAgent, env_name, config:dict, variant):
    base_env = gym.make(env_name)
    env = MountainCarRewardWrapper(base_env, variant=variant)
    
    n_episodes = config.get("n_episodes", 500)
    max_steps = config.get("max_steps", 1000)
    is_dqn = hasattr(agent, "memory")
    logs = []

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if is_dqn:
                agent.memory.push(state, action, reward, next_state, float(done))
                agent.learn()
            else:
                agent.store_reward(reward)

            state = next_state
            total_reward += reward
            if done:
                break

        if is_dqn:
            agent.decay_epsilon()
            if ep % config.get("target_update_freq", 10) == 0:
                agent.update_target()
        else:
            agent.learn()

        import numpy as np
        all_rewards = [row["reward"] for row in logs] + [total_reward]
        avg_100 = np.mean(all_rewards[-100:])
        logs.append({"episode": ep, "reward": total_reward, "avg_100": avg_100})

        if ep % 50 == 0:
            print(f"[{variant}] Ep {ep:>4}/{n_episodes} | Nagroda Shaped: {total_reward:>8.1f} | Avg100: {avg_100:>8.1f}")

    env.close()
    return logs

def main():
    variants = ["variant_A", "variant_B", "variant_C", "variant_D", "variant_E"]
    agents = ["DQN", "REINFORCE", "PPO"]
    
    # Słowniki na wyniki pogrupowane według algorytmów
    dqn_results = {}
    reinforce_results = {}
    ppo_results = {}

    for agent_name in agents:
        for variant in variants:
            print(f"\n=== URUCHAMIAM: {agent_name} z {variant} ===")
            
            if agent_name == "DQN":
                agent = DQNAgent(2, 3, CONFIGS["DQN"])
                config = CONFIGS["DQN"]
            elif agent_name == "REINFORCE":
                agent = REINFORCEAgent(2, 3, CONFIGS["REINFORCE"])
                config = CONFIGS["REINFORCE"]
            else:
                agent = PPOAgent(2, 3, CONFIGS["PPO"])
                config = CONFIGS["PPO"]
                
            logs = train_with_shaping(agent, ENV_NAME, config, variant)
            
            filename = f"{agent_name}_{ENV_NAME}_{variant}.csv"
            save_logs(logs, save_dir="results/logs", filename=filename)
            
            print(f"--- Ewaluacja {agent_name} ({variant}) ---")
            evaluate(agent, ENV_NAME, n_episodes=20)
            
            # Zbieranie czystych serii nagród do odpowiedniego słownika
            rewards = [row["reward"] for row in logs]
            
            if agent_name == "DQN":
                dqn_results[f"DQN_{variant}"] = rewards
            elif agent_name == "REINFORCE":
                reinforce_results[f"REINFORCE_{variant}"] = rewards
            else:
                ppo_results[f"PPO_{variant}"] = rewards

    # --- GENEROWANIE DWÓCH OSOBNYCH WYKRESÓW ---
    print("\n=== Generowanie wykresów porównawczych ===")
    
    # 1. Wykres dla DQN (pokaże warianty A, B, C obok siebie dla jednego algorytmu)
    plot_rewards(dqn_results, f"{ENV_NAME}_DQN_shaping", save_dir="plots")
    
    # 2. Wykres dla REINFORCE (pokaże warianty A, B, C dla drugiego algorytmu)
    plot_rewards(reinforce_results, f"{ENV_NAME}_REINFORCE_shaping", save_dir="plots")

    plot_rewards(ppo_results, f"{ENV_NAME}_PPO_shaping", save_dir="plots")

    print("\nWykresy zostały zapisane w folderze 'plots/'.")

if __name__ == "__main__":
    main()