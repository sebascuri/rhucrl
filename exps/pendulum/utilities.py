"""Python Script Template."""

import numpy as np
import torch
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from rllib.model import AbstractModel
from rllib.reward.state_action_reward import StateActionReward

from rhucrl.environment.wrappers import AdversarialWrapper


def pendulum_reset(wrapper, **kwargs):
    """Reset the pendulum."""
    high = np.array([np.pi, 0])
    wrapper.env.unwrapped.state = wrapper.env.np_random.uniform(low=high, high=high)
    wrapper.env.unwrapped.last_u = None
    return wrapper.env.unwrapped._get_obs()


class PendulumModel(AbstractModel):
    """Pendulum Model."""

    def __init__(self, alpha, attack_mode="gravity"):
        super().__init__(dim_state=(3,), dim_action=(3,))
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

        action = action[..., 0]

        g = 10.0
        m = 1.0
        length = 1.0
        dt = 0.05
        if self.attack_mode == "gravity":
            g = 10.0 * (1 + action[..., 1])
        else:
            m = 1.0 * (1 + action[..., 2])

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


class PendulumV1Env(PendulumEnv):
    """Other Pendulum overrides step method of pendulum.

    It uses properties intead of hard-coded values.
    """

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


class AdversarialPendulumWrapper(AdversarialWrapper):
    """Adversarial Pendulum Wrapper."""

    attacks = ["mass", "gravity"]

    def __init__(self, env, alpha=0.5, attack_mode="mass", **kwargs):
        antagonist_bounds = np.ones((2,))
        super().__init__(
            env=env,
            antagonist_low=-antagonist_bounds,
            antagonist_high=antagonist_bounds,
            alpha=alpha,
        )
        if attack_mode not in self.attacks:
            raise ValueError(f"{attack_mode} not in {self.attacks}.")
        self.attack_mode = attack_mode

    def adversarial_step(self, original_action, antagonist_action):
        """See AdversarialWrapper.step()."""
        if self.attack_mode == "gravity":
            self.env.g = 10.0 * (1 + antagonist_action[0])
        else:
            self.env.m = 1.0 * (1 + antagonist_action[1])
        return self.env.step(original_action)
