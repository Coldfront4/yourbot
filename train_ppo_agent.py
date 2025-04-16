import os
import logging
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from logger import get_logger

logger = get_logger("train_ppo_agent")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class RealisticTradingEnv(gym.Env):
    """
    Simulated trading environment using geometric Brownian motion, transaction costs, and reward shaping.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(RealisticTradingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        self.initial_price = 100.0
        self.mu = 0.0002
        self.sigma = 0.01
        self.commission_rate = 0.001
        self.max_steps = 200
        self.current_step = 0
        self.initial_cash = 10000.0
        self.cash = None
        self.position = None
        self.stock_price = None
        self.seed(RANDOM_SEED)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_step = 0
        self.stock_price = self.initial_price
        self.cash = self.initial_cash
        self.position = 0
        return self._get_obs()

    def step(self, action):
        prev_value = self._get_portfolio_value()
        dt = 1
        price_change = (self.mu * self.stock_price * dt +
                        self.sigma * self.stock_price * self.np_random.randn() * np.sqrt(dt))
        self.stock_price = max(0.01, self.stock_price + price_change)
        if action == 1:  # Buy
            cost = self.stock_price * (1 + self.commission_rate)
            if self.cash >= cost:
                self.cash -= cost
                self.position += 1
        elif action == 2:  # Sell
            if self.position > 0:
                proceeds = self.stock_price * (1 - self.commission_rate)
                self.cash += proceeds
                self.position -= 1

        self.current_step += 1
        done = self.current_step >= self.max_steps
        current_value = self._get_portfolio_value()
        profit = current_value - prev_value
        base_reward = profit / self.initial_cash
        risk_penalty = 0.0001 * (self.position ** 2)
        time_penalty = 0.00005 * self.current_step * abs(self.position)
        reward = base_reward - risk_penalty - time_penalty
        reward += 0.01 if profit > 0 else (-0.01 if profit < 0 else 0)
        return self._get_obs(), reward, done, {"portfolio_value": current_value, "cash": self.cash, "position": self.position}

    def _get_obs(self):
        value = self._get_portfolio_value()
        return np.array([self.stock_price / self.initial_price, self.position, value / self.initial_cash, self.current_step / self.max_steps], dtype=np.float32)

    def _get_portfolio_value(self):
        return self.cash + self.position * self.stock_price

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Price: {self.stock_price:.2f}, Position: {self.position}, Cash: {self.cash:.2f}, Portfolio: {self._get_portfolio_value():.2f}")

    def close(self):
        pass

if __name__ == "__main__":
    env = RealisticTradingEnv()
    model = PPO("MlpPolicy", env, verbose=1, device="cpu", learning_rate=1e-4,
                n_steps=256, batch_size=64, seed=RANDOM_SEED)
    total_timesteps = 50000
    logger.info(f"Starting PPO training for {total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=total_timesteps)
        model.save("ppo_rl_agent.zip")
        logger.info("Trained PPO agent saved as ppo_rl_agent.zip")
    except Exception as e:
        logger.error(f"Error during training: {e}")
