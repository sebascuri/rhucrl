"""Hopper Utility experiments."""

import numpy as np
import torch
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from rllib.reward.state_action_reward import StateActionReward

from exps.utilities import LargeStateTermination, healty_state

HEALTHY_REWARD = 5.0
FORWARD_REWARD_WEIGHT = 1.25
ACTION_COST = 0.1
HEALTHY_Z_RANGE = (1.0, 2.0)


class HumanoidV4Env(HumanoidEnv):
    """Humanoid Environment."""

    def __init__(self, action_cost=ACTION_COST):
        self.prev_pos = np.zeros(2)
        super().__init__(
            ctrl_cost_weight=action_cost,
            forward_reward_weight=FORWARD_REWARD_WEIGHT,
            contact_cost_weight=0.0,
            contact_cost_range=(-np.inf, 10.0),
            healthy_reward=HEALTHY_REWARD,
            terminate_when_unhealthy=True,
            healthy_z_range=HEALTHY_Z_RANGE,
            reset_noise_scale=1e-2,
        )

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        xy_velocity = (position[:2] - self.prev_pos) / self.dt
        self.prev_pos = position[:2]

        return np.concatenate((xy_velocity, position[2:], velocity)).ravel()


class HumanoidReward(StateActionReward):
    """Humanoid Reward."""

    dim_action = (17,)

    def __init__(self, action_cost_ratio=ACTION_COST, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio)

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        return FORWARD_REWARD_WEIGHT * state[..., 0] + HEALTHY_REWARD


class HumanoidTermination(LargeStateTermination):
    """Humanoid Termination Function."""

    @staticmethod
    def is_healthy(state):
        """Check if humanoid is healthy."""
        z = state[..., 2]
        return healty_state(z, healthy_range=HEALTHY_Z_RANGE)
