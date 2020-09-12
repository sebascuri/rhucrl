"""Swimmer Utility experiments."""

import numpy as np
import torch
from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from rllib.reward.state_action_reward import StateActionReward

from exps.utilities import LargeStateTermination

FORWARD_REWARD_WEIGHT = 1
ACTION_COST = 1e-4


class SwimmerV4Env(SwimmerEnv):
    """Swimmer Environment."""

    def __init__(self, action_cost=ACTION_COST):
        self.prev_pos = np.zeros(2)
        super().__init__(
            ctrl_cost_weight=action_cost, forward_reward_weight=FORWARD_REWARD_WEIGHT
        )

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        xy_velocity = (position[:2] - self.prev_pos) / self.dt
        self.prev_pos = position[:2]

        return np.concatenate((xy_velocity, position[2:], velocity)).ravel()


class SwimmerReward(StateActionReward):
    """Swimmer Reward."""

    dim_action = (2,)

    def __init__(self, action_cost_ratio=ACTION_COST, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio)

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        return FORWARD_REWARD_WEIGHT * state[..., 0]


SwimmerTermination = LargeStateTermination
