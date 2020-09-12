"""Inverted Pendulum Utility experiments."""

import torch
from rllib.reward.state_action_reward import StateActionReward

from exps.utilities import LargeStateTermination, healty_state

FORWARD_REWARD_WEIGHT = 1.0
ACTION_COST = 0
HEALTHY_ANGLE_RANGE = (-0.2, 0.2)


class PendulumReward(StateActionReward):
    """Inverted Pendulum Reward."""

    dim_action = (1,)

    def __init__(self, action_cost_ratio=ACTION_COST, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio)

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        return 1.0


class PendulumTermination(LargeStateTermination):
    """Inverted Pendulum Termination Function."""

    @staticmethod
    def is_healthy(state):
        """Check if pendulum angle is healthy."""
        angle = state[..., 1]
        return healty_state(angle, healthy_range=HEALTHY_ANGLE_RANGE)
