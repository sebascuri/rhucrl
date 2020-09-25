"""Python Script Template."""

import numpy as np
import torch
from rllib.model import AbstractModel


class PendulumModel(AbstractModel):
    """Pendulum Model."""

    def __init__(
        self, alpha, force_body_names=("mass",), wrapper="adversarial_pendulum"
    ):
        if alpha == 0:
            dim_action = (1,)
        elif wrapper == "adversarial_pendulum":
            dim_action = (1 + len(force_body_names),)
        else:
            dim_action = (2,)
        super().__init__(dim_state=(3,), dim_action=dim_action)
        self.max_speed = 8
        self.max_torque = 2.0
        self.alpha = alpha
        self.force_body_names = {name: i for i, name in enumerate(force_body_names)}
        self.wrapper = wrapper

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def forward(self, state, action, next_state=None):
        """Compute Next State distribution."""
        theta, omega = torch.atan2(state[..., 1], state[..., 0]), state[..., 2]

        protagonist_action = (
            action[..., 0].clone().clamp(-self.max_torque, self.max_torque)
        )
        antagonist_action = action[..., 1:]
        g = 10.0
        m = 1.0
        length = 1.0
        dt = 0.05

        if self.alpha == 0:
            u = protagonist_action
        elif self.wrapper == "noisy_action":
            antagonist_action = antagonist_action[..., 0].clamp(
                -self.max_torque, self.max_torque
            )
            u = (1 - self.alpha) * protagonist_action + self.alpha * antagonist_action
        elif self.wrapper == "probabilistic_action":
            antagonist_action = antagonist_action[..., 0].clamp(
                -self.max_torque, self.max_torque
            )
            if state.ndim == 1:
                if np.random.rand() < self.alpha:
                    u = antagonist_action
                else:
                    u = protagonist_action
            else:
                u = protagonist_action.clone()
                idx = np.random.rand(state.shape[0]) < self.alpha
                u[idx] = antagonist_action[idx].clone()

        elif self.wrapper == "adversarial_pendulum" and self.alpha > 0:
            u = torch.clamp(protagonist_action, -self.max_torque, self.max_torque)
            if "gravity" in self.force_body_names:
                idx = self.force_body_names["gravity"]
                g = g * (1 + antagonist_action[..., idx])

            if "mass" in self.force_body_names:
                idx = self.force_body_names["mass"]
                m = m * (1 + antagonist_action[..., idx])
        else:
            raise NotImplementedError
        inertia = (m * length ** 2) / 3.0
        omega_dot = -3 * g / (2 * length) * torch.sin(theta + np.pi) + u / inertia
        new_omega = omega + omega_dot * dt
        new_theta = theta + new_omega * dt
        new_omega = torch.clamp(new_omega, -self.max_speed, self.max_speed)

        return (
            torch.stack(
                (torch.cos(new_theta), torch.sin(new_theta), new_omega), dim=-1
            ),
            torch.tensor(0.0),
        )
