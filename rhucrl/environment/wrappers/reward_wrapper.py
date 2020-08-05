"""Reward Wrapper."""
from typing import Callable, Tuple

import numpy as np
from gym import Env, Wrapper


class RewardWrapper(Wrapper):
    """Wrap environment by changing the reward function."""

    def __init__(
        self,
        env: Env,
        reward_function: Callable[[np.ndarray, np.ndarray, np.ndarray, dict], float],
    ):
        super().__init__(env)
        self.reward_function = reward_function

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Wrap reward function."""
        obs = self.env.unwrapped._get_obs()
        next_obs, _, done, info = self.env.step(action)
        reward = self.reward_function(obs, action, next_obs, info)
        return next_obs, reward, done, info
