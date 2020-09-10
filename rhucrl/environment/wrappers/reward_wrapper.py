"""Reward Wrapper."""
from gym import Wrapper


class RewardWrapper(Wrapper):
    """Wrap environment by changing the reward function."""

    def __init__(self, env, reward_function):
        super().__init__(env)
        self.reward_function = reward_function

    def step(self, action):
        """Wrap reward function."""
        obs = self.env.unwrapped._get_obs()
        next_obs, _, done, info = self.env.step(action)
        reward = self.reward_function(obs, action, next_obs, info)
        return next_obs, reward, done, info
