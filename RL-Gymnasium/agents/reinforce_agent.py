import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):

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
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class REINFORCEAgent:

    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(
            state_dim, action_dim, config.get("hidden_dims", [128, 128])
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=config.get("lr", 1e-3)
        )
        self.gamma = config.get("gamma", 0.99)

        # Historia epizodu
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self):
        # Zdyskontowane zwroty G_t
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalizacja zwrotów
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Straty policy gradient
        policy_loss = []
        for log_prob, G_t in zip(self.log_probs, returns_t):
            policy_loss.append(-log_prob * G_t)

        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10)
        self.optimizer.step()

        # Wyczyść historię epizodu
        self.log_probs = []
        self.rewards = []

        return loss.item()

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"[REINFORCE] Model zapisany: {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[REINFORCE] Model wczytany: {path}")
