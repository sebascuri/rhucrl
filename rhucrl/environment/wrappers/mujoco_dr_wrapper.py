"""Domain Randomization Wrapper."""

import numpy as np

from .adversarial_wrapper import AdversarialWrapper


class MujocoDomainRandomizationWrapper(AdversarialWrapper):
    """
    Wrapper for Mujoco domain randomization environments.

    By setting the antagonist action, the mass and friction coefficients are changed.
    """

    eps = 0.001

    def __init__(self, env, mass_names=None, friction_names=None):
        # Change mass.
        mass_names = [] if mass_names is None else mass_names
        self.mass_names = {
            name: (
                env.model.body_names.index(name),
                env.model.body_mass[env.model.body_names.index(name)],
            )
            for name in mass_names
        }

        # Change friction coefficient.
        friction_names = [] if friction_names is None else friction_names
        self.friction_names = {
            name: (
                env.model.body_names.index(name),
                env.model.geom_friction[env.model.body_names.index(name)],
            )
            for name in friction_names
        }

        size = len(mass_names) + len(friction_names)
        antagonist_high = np.ones(size)
        antagonist_low = -np.ones(size)

        super().__init__(
            env=env,
            antagonist_low=antagonist_low,
            antagonist_high=antagonist_high,
            alpha=1.0,
        )

    def _antagonist_action_to_mass(self, antagonist_action):
        for i, (body_name, (idx, base_mass)) in enumerate(self.mass_names.items()):
            new_mass = (1 + antagonist_action[i] + self.eps) * base_mass
            self.env.model.body_mass[idx] = new_mass

    def _antagonist_action_to_friction(self, antagonist_action):
        for i, (body_name, (idx, base_friction)) in enumerate(
            self.friction_names.items()
        ):
            new_friction = (1 + antagonist_action[i] + self.eps) * base_friction
            self.env.model.geom_friction[idx] = new_friction

    def adversarial_step(self, protagonist_action, antagonist_action):
        """See `AdversarialWrapper.adversarial_step()'."""
        self._antagonist_action_to_mass(antagonist_action[: len(self.mass_names)])
        self._antagonist_action_to_friction(antagonist_action[len(self.mass_names) :])
        return self.env.step(protagonist_action)

    @property
    def name(self):
        """Get wrapper name."""
        return "Domain Randomization " + "-".join(self.force_body_names)
