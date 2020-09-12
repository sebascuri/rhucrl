"""Walker-2d Utility experiments."""

import numpy as np
import torch
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
from rllib.reward.state_action_reward import StateActionReward

from exps.utilities import LargeStateTermination, healty_state

HEALTHY_REWARD = 1.0
FORWARD_REWARD_WEIGHT = 1.0
ACTION_COST = 1e-3
HEALTHY_Z_RANGE = (0.8, 2.0)
HEALTHY_ANGLE_RANGE = (-1.0, 1.0)
HEALTHY_STATE_RANGE = (100.0, 100.0)


class Walker2dV4Env(Walker2dEnv):
    """Walker2d Environment."""

    def __init__(self, action_cost=ACTION_COST):
        self.prev_pos = np.zeros(1)
        super().__init__(
            forward_reward_weight=FORWARD_REWARD_WEIGHT,
            ctrl_cost_weight=action_cost,
            healthy_reward=HEALTHY_REWARD,
            terminate_when_unhealthy=True,
            healthy_z_range=HEALTHY_Z_RANGE,
            healthy_angle_range=(-1.0, 1.0),
        )

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)

        x_velocity = (position[:1] - self.prev_pos) / self.dt
        self.prev_pos = position[:1]

        return np.concatenate((x_velocity, position[2:], velocity)).ravel()


class Walker2dReward(StateActionReward):
    """Walker2d Reward."""

    dim_action = (6,)

    def __init__(self, action_cost_ratio=ACTION_COST, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio)

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        return FORWARD_REWARD_WEIGHT * state[..., 0] + FORWARD_REWARD_WEIGHT


class Walker2dTermination(LargeStateTermination):
    """Walker2d Termination Function."""

    @staticmethod
    def is_healthy(state):
        """Check if walker is healthy."""
        z = state[..., 1]
        angle = state[..., 2]

        healthy_state = healty_state(state[..., 2:], healthy_range=HEALTHY_STATE_RANGE)
        healthy_z = healty_state(z, healthy_range=HEALTHY_Z_RANGE)
        healthy_angle = healty_state(angle, healthy_range=HEALTHY_ANGLE_RANGE)

        return healthy_state.all(-1) * healthy_angle * healthy_z
