"""Adversarial Pendulum."""

import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize

from .adversarial_wrapper import AdversarialWrapper


class OtherPendulum(PendulumEnv):
    """Other Pendulum overrides step method of pendulum."""

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


class PendulumAdvEnv(AdversarialWrapper):
    """Adversarial Pendulum Environment."""

    def __init__(self, alpha=0.5, attack_mode="mass"):
        antagonist_bounds = 2 * np.ones((1,))
        super().__init__(
            env=OtherPendulum(g=10.0),
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
            self.env.g = 10.0 + antagonist_action[0]
        else:
            self.env.m = 1.0 + antagonist_action[0]
        return self.env.step(original_action)
