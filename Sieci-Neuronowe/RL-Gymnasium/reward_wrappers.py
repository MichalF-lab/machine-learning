import gymnasium as gym
import numpy as np

class MountainCarRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, variant="baseline"):
        super().__init__(env)
        self.variant = variant
        self.gamma = 0.99 # taka sama jak w konfiguracji agenta
        self.max_position_in_episode = -1.2 # potrzebne do wariantu E

    def reset(self, **kwargs):
        self.max_position_in_episode = -1.2 # resetujemy licznik przy nowym epizodzie
        return self.env.reset(**kwargs)

    def reward(self, reward):
        state = self.env.unwrapped.state 
        position, velocity = state[0], state[1]
                
        if self.variant == "baseline":
            return reward
            
        elif self.variant == "variant_A": # Bonus za odsunięcie się od środka
            bonus = (position - (-0.5)) * 10.0 if position > -0.5 and velocity > 0 else 0.0
            return reward + bonus
            
        elif self.variant == "variant_B": # Bonus za prędkość 
            
            raw_bonus = abs(velocity) * 10.0  
            bonus = np.clip(raw_bonus, 0.0, 0.8) 
            
            return reward + bonus

        elif self.variant == "variant_C": # Hybryda A i B
            bonus_pos = (position - (-0.5)) * 0.5 if position > -0.5 else 0.0
            bonus_vel = abs(velocity) * 5.0
            
            total_bonus = np.clip(bonus_pos + bonus_vel, 0.0, 0.9)
            
            return reward + total_bonus
            
        elif self.variant == "variant_D":
            # Potential-Based Reward Shaping
            # Wysokość w MountainCar jest funkcją sinusa pozycji
            height = np.sin(3 * position)
            
            # Potencjał obecnego stanu (Energia potencjalna + kinetyczna z wagami)
            potential_next = 100.0 * height + 500.0 * (velocity ** 2)
            
            # Odtwarzamy poprzedni stan w przybliżeniu, żeby obliczyć jego potencjał
            prev_position = position - velocity
            prev_height = np.sin(3 * prev_position)
            potential_prev = 100.0 * prev_height + 500.0 * ((velocity) ** 2) # uproszczenie dla v
            
            # Formuła PBRS: gamma * Phi(s') - Phi(s)
            pbrs_bonus = (self.gamma * potential_next) - potential_prev
            return reward + pbrs_bonus
            
        elif self.variant == "variant_E":
            # Bonus za nowy rekord wysokości
            bonus = 0.0
            if position > self.max_position_in_episode:
                bonus = (position - self.max_position_in_episode) * 50.0
                self.max_position_in_episode = position
            return reward + bonus
            
        return reward