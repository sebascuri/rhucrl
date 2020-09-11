"""Python Script Template."""

import numpy as np

from .adversarial_wrapper import AdversarialWrapper


class MujocoAdversarialWrapper(AdversarialWrapper):
    """Wrapper for Mujoco adversarial environments."""

    def __init__(
        self, env, alpha=5.0, force_body_names=None, new_mass=None, new_friction=None
    ):
        force_body_names = [] if force_body_names is None else force_body_names
        self._antagonist_bindex = [
            env.model.body_names.index(i) for i in force_body_names
        ]
        antagonist_high = np.ones(2 * len(force_body_names))
        antagonist_low = -antagonist_high

        new_mass = {} if new_mass is None else new_mass
        for body_name, weight in new_mass.items():
            env.model.body_mass[env.model.body_names.index(body_name)] = weight
        new_friction = {} if new_friction is None else new_friction
        for body_name, friction in new_friction.items():
            env.model.geom_friction[env.model.body_names.index(body_name), 0] = friction

        super().__init__(
            env=env,
            antagonist_low=antagonist_low,
            antagonist_high=antagonist_high,
            alpha=alpha,
        )

    def _antagonist_action_to_xfrc(self, antagonist_action):
        aa = antagonist_action
        for i, bindex in enumerate(self._antagonist_bindex):
            self.sim.data.xfrc_applied[bindex] = np.array(
                [aa[i * 2], 0.0, aa[i * 2 + 1], 0.0, 0.0, 0.0]
            )

    def adversarial_step(self, protagonist_action, antagonist_action):
        """See `AdversarialWrapper.adversarial_step()'."""
        self._antagonist_action_to_xfrc(antagonist_action)
        return self.env.step(protagonist_action)
