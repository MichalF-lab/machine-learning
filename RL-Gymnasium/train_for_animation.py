"""
Trenuje DQN + Wariant D (PBRS) na MountainCar-v0 (te same hiperparametry co
run_shaping_experiments.py) i po drodze nagrywa trajektorie (pozycja/prędkość)
polityki agenta w kilku punktach treningu, żeby pokazać jak auto uczy się
wjeżdżać na wzgórze.

Wyjscie: D:\\CLCO\\Projekt_3\\animacje\\car_trajectories.json
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import numpy as np

from agents.dqn_agent import DQNAgent
from reward_wrappers import MountainCarRewardWrapper

ENV_NAME = "MountainCar-v0"
GOAL_POSITION = 0.5
CHECKPOINT_EPISODES = list(range(0, 501, 25))

DQN_CONFIG = {
    "n_episodes": 500, "max_steps": 1000, "lr": 1e-3, "gamma": 0.99,
    "epsilon_start": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.995,
    "batch_size": 64, "buffer_size": 100_000, "target_update_freq": 10, "hidden_dims": [128, 128],
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "animacje")
os.makedirs(OUT_DIR, exist_ok=True)


def record_rollout(agent, env_name, greedy=True, random_policy=False, max_steps=200):
    env = gym.make(env_name)
    state, _ = env.reset(seed=42)
    positions, velocities = [float(state[0])], [float(state[1])]

    saved_epsilon = agent.epsilon
    if greedy:
        agent.epsilon = 0.0

    solved = False
    steps = 0
    for _ in range(max_steps):
        if random_policy:
            action = random.randrange(agent.action_dim)
        else:
            action = agent.select_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        positions.append(float(state[0]))
        velocities.append(float(state[1]))
        steps += 1
        if terminated:
            solved = True
            break
        if truncated:
            break

    agent.epsilon = saved_epsilon
    env.close()
    return {"positions": positions, "velocities": velocities, "solved": solved, "steps": steps}


def main():
    state_dim, action_dim = 2, 3
    agent = DQNAgent(state_dim, action_dim, DQN_CONFIG)

    base_env = gym.make(ENV_NAME)
    env = MountainCarRewardWrapper(base_env, variant="variant_D")

    n_episodes = DQN_CONFIG["n_episodes"]
    max_steps = DQN_CONFIG["max_steps"]
    target_update_freq = DQN_CONFIG.get("target_update_freq", 10)

    trajectories = {}
    trajectories["0"] = record_rollout(agent, ENV_NAME, random_policy=True)
    print(f"[checkpoint 0 / random] solved={trajectories['0']['solved']} steps={trajectories['0']['steps']}")

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, float(done))
            agent.learn()

            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        if ep % target_update_freq == 0:
            agent.update_target()

        if ep % 50 == 0:
            print(f"Ep {ep:>4}/{n_episodes} | Nagroda: {total_reward:>8.1f} | epsilon={agent.epsilon:.3f}")

        if ep in CHECKPOINT_EPISODES:
            traj = record_rollout(agent, ENV_NAME, greedy=True)
            trajectories[str(ep)] = traj
            print(f"[checkpoint {ep}] solved={traj['solved']} steps={traj['steps']} "
                  f"max_pos={max(traj['positions']):.3f}")

    agent.save(os.path.join("results", "models", "DQN_MountainCar-v0_variant_D.pt"))

    with open(os.path.join(OUT_DIR, "car_trajectories.json"), "w", encoding="utf-8") as f:
        json.dump(trajectories, f)
    print("Zapisano trajektorie:", os.path.join(OUT_DIR, "car_trajectories.json"))


if __name__ == "__main__":
    main()
