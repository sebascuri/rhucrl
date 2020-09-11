"""Half Cheetah Utility experiments.."""

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from rllib.reward.state_action_reward import StateActionReward

from rhucrl.environment.wrappers import AdversarialWrapper


class HalfCheetahReward(StateActionReward):
    """Get Pendulum Reward."""

    dim_action = (6,)

    def __init__(self, action_cost_ratio=0.1, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio)

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        return state[..., 0]


class HalfCheetahV4Env(HalfCheetahEnv):
    """Other Pendulum overrides step method of pendulum.

    It uses properties intead of hard-coded values.
    """

    def __init__(self, action_cost=0.1):
        self.prev_x_pos = 0.0
        super().__init__(ctrl_cost_weight=action_cost)

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        forward_velocity = (position[0] - self.prev_x_pos) / self.dt
        self.prev_x_pos = position[0]

        return np.concatenate(([forward_velocity], position[1:], velocity)).ravel()
