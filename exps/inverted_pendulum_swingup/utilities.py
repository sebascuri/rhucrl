"""Python Script Template."""

import numpy as np
import torch
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from rllib.model import AbstractModel
from rllib.reward.state_action_reward import StateActionReward


class PendulumV1Env(PendulumEnv):
    """Other Pendulum overrides step method of pendulum.

    It uses properties intead of hard-coded values.
    """

    def reset(self):
        """Reset to fix initial conditions."""
        # high = np.array([np.pi, 1])
        # self.state = self.np_random.uniform(low=-high, high=high)
        high = np.array([np.pi, 0])
        self.state = self.np_random.uniform(low=high, high=high)
        self.last_u = None
        return self._get_obs()

    def step(self, u):
        """Override step method of pendulum env."""
        th, thdot = self.state

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        i = self.m * self.l ** 2
        newthdot = (
            thdot
            + (-3 * self.g / (2 * self.l) * np.sin(th + np.pi) + 3.0 / i * u) * self.dt
        )
        newth = th + newthdot * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}


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


class PendulumModel(AbstractModel):
    """Pendulum Model."""

    def __init__(self, alpha, force_body_names=("mass",)):
        super().__init__(dim_state=(3,), dim_action=(1 + len(force_body_names),))
        self.max_speed = 8
        self.max_torque = 2.0
        self.alpha = alpha
        self.force_body_names = {name: i for i, name in enumerate(force_body_names)}

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def forward(self, state, action, next_state=None):
        """Compute Next State distribution."""
        th, thdot = torch.atan2(state[..., 1], state[..., 0]), state[..., 2]

        protagonist_action = action[..., 0]
        u = torch.clamp(protagonist_action, -self.max_torque, self.max_torque)

        antagonist_action = action[..., 1:]
        g = 10.0
        m = 1.0
        length = 1.0
        dt = 0.05

        if "gravity" in self.force_body_names:
            idx = self.force_body_names["gravity"]
            g = 10.0 * (1 + antagonist_action[..., idx])

        if "mass" in self.force_body_names:
            idx = self.force_body_names["mass"]
            m = 1.0 * (1 + antagonist_action[..., idx])

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
