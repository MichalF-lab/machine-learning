import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    # Sieć neuronowa aproksymująca funkcję wartości Q(s, a).

    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super().__init__()

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    # Experience Replay Buffer – przechowuje przejścia (s, a, r, s', done).

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(self, state_dim, action_dim, config):
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sieci
        self.policy_net = QNetwork(
            state_dim, action_dim, config.get("hidden_dims", [128, 128])
        ).to(self.device)
        self.target_net = QNetwork(
            state_dim, action_dim, config.get("hidden_dims", [128, 128])
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer + loss
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.get("lr", 1e-3)
        )
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(config.get("buffer_size", 100_000))

        # Epsilon (eksploracja)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

        # Liczniki
        self.steps = 0
        self.target_update_freq = config.get("target_update_freq", 10)

    def select_action(self, state):
        # eksploracja vs eksploatacja
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()

    def learn(self):
        batch_size = self.config.get("batch_size", 64)
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a) z policy net
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # max Q(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target = rewards_t + self.config.get("gamma", 0.99) * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)
        print(f"[DQN] Model zapisany: {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        print(f"[DQN] Model wczytany: {path}")
