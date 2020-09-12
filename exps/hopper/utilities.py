"""Hopper Utility experiments."""

import numpy as np
import torch
from gym.envs.mujoco.hopper_v3 import HopperEnv
from rllib.reward.state_action_reward import StateActionReward

from exps.utilities import LargeStateTermination, healty_state

HEALTHY_REWARD = 1.0
FORWARD_REWARD_WEIGHT = 1.25
ACTION_COST = 1e-3
HEALTHY_Z_RANGE = (0.7, float("inf"))
HEALTHY_ANGLE_RANGE = (-0.2, 0.2)
HEALTHY_STATE_RANGE = (-100.0, 100.0)


class HopperV4Env(HopperEnv):
    """Hopper Environment."""

    def __init__(self, action_cost=ACTION_COST):
        self.prev_pos = np.zeros(2)
        super().__init__(
            ctrl_cost_weight=action_cost,
            forward_reward_weight=FORWARD_REWARD_WEIGHT,
            healthy_reward=HEALTHY_REWARD,
            terminate_when_unhealthy=True,
            healthy_state_range=(-100.0, 100.0),
            healthy_z_range=HEALTHY_Z_RANGE,
            healthy_angle_range=HEALTHY_ANGLE_RANGE,
        )

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        xy_velocity = (position[:2] - self.prev_pos) / self.dt
        self.prev_pos = position[:2]

        return np.concatenate((xy_velocity, position[2:], velocity)).ravel()


class HopperReward(StateActionReward):
    """Hopper Reward."""

    dim_action = (3,)

    def __init__(self, action_cost_ratio=ACTION_COST, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio)

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        return FORWARD_REWARD_WEIGHT * state[..., 0] + HEALTHY_REWARD


class HopperTermination(LargeStateTermination):
    """Hopper Termination Function."""

    @staticmethod
    def is_healthy(state):
        """Check if hopper is healthy."""
        z = state[..., 1]
        angle = state[..., 2]

        healthy_state = healty_state(state[..., 2:], healthy_range=HEALTHY_STATE_RANGE)
        healthy_z = healty_state(z, healthy_range=HEALTHY_Z_RANGE)
        healthy_angle = healty_state(angle, healthy_range=HEALTHY_ANGLE_RANGE)

        return healthy_state.all(-1) * healthy_angle * healthy_z
