"""Half Cheetah Utility experiments."""

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from rllib.reward.state_action_reward import StateActionReward

from exps.utilities import LargeStateTermination

FORWARD_REWARD_WEIGHT = 1.0
ACTION_COST = 0.1


class HalfCheetahV4Env(HalfCheetahEnv):
    """Half-Cheetah Environment."""

    def __init__(self, action_cost=ACTION_COST):
        self.prev_pos = np.zeros(1)
        super().__init__(
            ctrl_cost_weight=action_cost, forward_reward_weight=FORWARD_REWARD_WEIGHT
        )

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        forward_velocity = (position[:1] - self.prev_pos) / self.dt
        self.prev_pos = position[:1]

        return np.concatenate((forward_velocity, position[1:], velocity)).ravel()


class HalfCheetahReward(StateActionReward):
    """Half-Cheetah Reward."""

    dim_action = (6,)

    def __init__(self, action_cost_ratio=ACTION_COST, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio)

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        return FORWARD_REWARD_WEIGHT * state[..., 0]


HalfCheetahTermination = LargeStateTermination
