"""Adversarial wrapper for pendulum."""
import numpy as np

from .adversarial_wrapper import AdversarialWrapper
from .mujoco_adversarial_wrapper import MujocoAdversarialWrapper


class AdversarialPendulumWrapper(MujocoAdversarialWrapper):
    """Adversarial Pendulum Wrapper."""

    def __init__(self, env, alpha=0.5, force_body_names=("mass", "gravity"), **kwargs):
        antagonist_bounds = np.ones((len(force_body_names),))
        AdversarialWrapper.__init__(
            self,
            env=env,
            antagonist_low=-antagonist_bounds,
            antagonist_high=antagonist_bounds,
            alpha=alpha,
        )
        self.force_body_names = {name: i for i, name in enumerate(force_body_names)}

    def _antagonist_action_to_xfrc(self, antagonist_action):
        if "gravity" in self.force_body_names:
            idx = self.force_body_names["gravity"]
            self.env.g = 10.0 * (1 + self.alpha * antagonist_action[idx])

        if "mass" in self.force_body_names:
            idx = self.force_body_names["mass"]
            self.env.m = 1.0 * (1 + self.alpha * antagonist_action[idx])
