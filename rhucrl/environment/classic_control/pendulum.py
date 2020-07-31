"""Adversarial Pendulum."""

from typing import Tuple

import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv

from rhucrl.environment.adversarial_wrapper import AdversarialWrapper


class AdversarialPendulumEnv(AdversarialWrapper):
    """Adversarial Pendulum Environment."""

    env: PendulumEnv

    def __init__(self, alpha=0.5):
        adversarial_bounds = 100 * np.ones((1,))
        super().__init__(
            env=PendulumEnv(g=10.0),
            adversarial_low=-adversarial_bounds,
            adversarial_high=adversarial_bounds,
            alpha=alpha,
        )

    def adversarial_step(
        self, original_action: np.ndarray, adversarial_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """See AdversarialWrapper.step()."""
        self.env.g = self.env.g + adversarial_action[0]

        return self.env.step(original_action)
