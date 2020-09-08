"""Python Script Template."""

import numpy as np
import torch
from rllib.model import AbstractModel
from rllib.reward.state_action_reward import StateActionReward
from rllib.util.utilities import get_backend


def pendulum_reward(state, action, next_state=None, info=None):
    """Calculate the pendulum reward."""
    bk = get_backend(state)
    th, thdot = bk.arctan2(state[..., 1], state[..., 0]), state[..., 2]
    action = action[..., 0]

    return -(th ** 2 + 0.1 * thdot ** 2 + 0.001 * (action ** 2))


def pendulum_reset(wrapper, **kwargs):
    """Reset the pendulum."""
    high = np.array([np.pi, 0])
    wrapper.env.unwrapped.state = wrapper.env.np_random.uniform(low=high, high=high)
    wrapper.env.unwrapped.last_u = None
    return wrapper.env.unwrapped._get_obs()


class PendulumModel(AbstractModel):
    """Pendulum Model."""

    def __init__(self, alpha, attack_mode="gravity"):
        super().__init__(dim_state=(3,), dim_action=(2,))
        self.max_speed = 8
        self.max_torque = 2.0
        self.alpha = alpha
        self.attack_mode = attack_mode

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def forward(self, state, action, next_state=None):
        """Compute Next State distribution."""
        th, thdot = torch.atan2(state[..., 1], state[..., 0]), state[..., 2]

        p_action = action[..., 0]
        a_action = action[..., 1]
        action = p_action

        g = 10.0
        m = 1.0
        length = 1.0
        dt = 0.05
        if self.attack_mode == "gravity":
            g += a_action
        else:
            m += a_action

        u = torch.clamp(action, -self.max_torque, self.max_torque)

        newthdot = (
            thdot
            + (
                -3 * g / (2 * length) * torch.sin(th + np.pi)
                + 3.0 / (m * length ** 2) * u
            )
            * dt
        )
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -self.max_speed, self.max_speed)

        return (
            torch.stack((torch.cos(newth), torch.sin(newth), newthdot), dim=-1),
            torch.tensor(0.0),
        )


class PendulumReward(StateActionReward):
    """Get Pendulum Reward."""

    dim_action = (1,)

    def __init__(self, action_cost_ratio=0.001, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio)

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        th, thdot = torch.atan2(state[..., 1], state[..., 0]), state[..., 2]
        return -(th ** 2 + 0.1 * thdot ** 2)
