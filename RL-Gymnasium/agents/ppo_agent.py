import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super().__init__()
        
        actor_layers = []
        in_dim = state_dim
        for h in hidden_dims:
            actor_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        critic_layers = []
        in_dim = state_dim
        for h in hidden_dims:
            critic_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, state):
        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1)
        state_value = self.critic(state)
        return probs, state_value


class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = config.get("gamma", 0.99)
        self.eps_clip = config.get("eps_clip", 0.2)
        self.K_epochs = config.get("K_epochs", 4)
        
        self.policy = ActorCriticNetwork(
            state_dim, action_dim, config.get("hidden_dims", [128, 128])
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.get("lr", 1e-3))
        self.loss_fn = nn.MSELoss()
        
        self.eps = 1e-8

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs, _ = self.policy(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self):
        if len(self.rewards) == 0:
            return None

        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
            
        old_states = torch.FloatTensor(np.array(self.states)).to(self.device)
        old_actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device).unsqueeze(1)

        with torch.no_grad():
            _, state_values = self.policy(old_states)
            advantages = returns_t - state_values
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

        for _ in range(self.K_epochs):
            probs, state_values = self.policy(old_states)
            dist = torch.distributions.Categorical(probs)
            
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()
            
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            advantages_flat = advantages.squeeze(1)
            
            surr1 = ratios * advantages_flat
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages_flat
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(state_values, returns_t)
            
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10)
            self.optimizer.step()

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        
        return total_loss.item()

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"[PPO] Model zapisany: {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[PPO] Model wczytany: {path}")