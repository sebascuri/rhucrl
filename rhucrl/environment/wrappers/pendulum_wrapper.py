"""Adversarial wrapper for pendulum."""
import numpy as np

from .adversarial_wrapper import AdversarialWrapper


class AdversarialPendulumWrapper(AdversarialWrapper):
    """Adversarial Pendulum Wrapper."""

    def __init__(self, env, alpha=0.5, force_body_names=("mass",), **kwargs):
        antagonist_bounds = np.ones((len(force_body_names),))
        super().__init__(
            env=env,
            antagonist_low=-antagonist_bounds,
            antagonist_high=antagonist_bounds,
            alpha=alpha,
        )
        self.force_body_names = {name: i for i, name in enumerate(force_body_names)}

    def adversarial_step(self, original_action, antagonist_action):
        """See AdversarialWrapper.step()."""
        if "gravity" in self.force_body_names:
            idx = self.force_body_names["gravity"]
            self.env.g = 10.0 * (1 + antagonist_action[idx])

        if "mass" in self.force_body_names:
            idx = self.force_body_names["mass"]
            self.env.m = 1.0 * (1 + antagonist_action[idx])
        return self.env.step(original_action)

    @property
    def name(self):
        """Get wrapper name."""
        return "Adversarial" + "-".join(self.force_body_names.keys())
