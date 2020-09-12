"""Ant Utility experiments."""

import numpy as np
import torch
from gym.envs.mujoco.ant_v3 import AntEnv
from rllib.reward.state_action_reward import StateActionReward

from exps.utilities import LargeStateTermination, healty_state

HEALTHY_REWARD = 1.0
FORWARD_REWARD_WEIGHT = 1.0
ACTION_COST = 0.5
HEALTHY_Z_RANGE = (0.2, 1.0)


class AntV4Env(AntEnv):
    """Ant Environment."""

    def __init__(self, action_cost=ACTION_COST):
        self.prev_pos = np.zeros(2)
        super().__init__(
            ctrl_cost_weight=ACTION_COST,
            contact_cost_weight=0,
            healthy_reward=HEALTHY_REWARD,
            terminate_when_unhealthy=True,
            healthy_z_range=HEALTHY_Z_RANGE,
            reset_noise_scale=0.1,
        )

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        xy_velocity = (position[:2] - self.prev_pos) / self.dt
        self.prev_pos = position[:2]

        return np.concatenate((xy_velocity, position[2:], velocity)).ravel()


class AntReward(StateActionReward):
    """Ant Reward Model."""

    dim_action = (17,)

    def __init__(self, action_cost_ratio=ACTION_COST, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio)

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        return FORWARD_REWARD_WEIGHT * state[..., 0] + HEALTHY_REWARD


class AntTermination(LargeStateTermination):
    """Ant Termination Function."""

    @staticmethod
    def is_healthy(state):
        """Check if ant is healthy."""
        z = state[..., 2]
        return healty_state(z, healthy_range=HEALTHY_Z_RANGE)
