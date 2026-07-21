import gymnasium as gym
import numpy as np

def train(agent, env_name, config):
    # Uniwersalna pętla treningowa (DQN lub REINFORCE)

    env = gym.make(env_name)
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

        # Aktualizacja po epizodzie
        if is_dqn:
            agent.decay_epsilon()
            if ep % config.get("target_update_freq", 10) == 0:
                agent.update_target()
        else:
            agent.learn()

        # Logi
        all_rewards = [row["reward"] for row in logs] + [total_reward]
        avg_100 = np.mean(all_rewards[-100:])
        logs.append({"episode": ep, "reward": total_reward, "avg_100": avg_100})

        if ep % 50 == 0:
            print(f"Ep {ep:>4}/{n_episodes} | Nagroda: {total_reward:>8.1f} | Avg100: {avg_100:>8.1f}")

    env.close()
    return logs


def evaluate(agent, env_name, n_episodes=20):
    # Ewaluacja bez eksploracji

    env = gym.make(env_name)

    saved_epsilon = getattr(agent, "epsilon", None)
    if saved_epsilon is not None:
        agent.epsilon = 0.0

    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)

    if saved_epsilon is not None:
        agent.epsilon = saved_epsilon

    env.close()
    print(f"[Ewaluacja] Średnia: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return rewards
