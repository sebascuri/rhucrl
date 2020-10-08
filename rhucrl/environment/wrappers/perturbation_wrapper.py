"""Python Script Template."""
from gym import Wrapper


class PerturbationWrapper(Wrapper):
    """Perturbation Wrapper."""

    def __init__(self, env, new_mass, new_friction=(), relative=True):
        super().__init__(env)
        self.relative = relative
        self.body_mass = {}
        for name in new_mass:
            idx = self.get_index(name)
            value = env.model.body_mass[idx].copy()
            self.body_mass.update(**{name: (idx, value)})

        self.body_friction = {}
        for name in new_friction:
            idx = self.get_index(name)
            value = env.model.geom_friction[idx, 0].copy()
            self.body_friction.update(**{name: (idx, value)})

    def get_index(self, name):
        """Get index of a given name."""
        try:
            idx = self.env.model.body_names.index(name)
        except ValueError:
            idx = self.env.model.body_names.index("r_forearm_link")
        return idx

    def update(self, new_mass, new_friction=None):
        """Update environment."""
        for name, new_mass_ in new_mass.items():
            idx, original_value = self.body_mass[name]
            if self.relative:
                self.env.model.body_mass[idx] = original_value * new_mass_
            else:
                self.env.model.body_mass[idx] = new_mass_

        if new_friction is None:
            return
        for name, new_friction_ in new_friction.items():
            idx, original_value = self.body_friction[name]
            if self.relative:
                self.env.model.geom_friction[idx, 0] = original_value * new_friction_
            else:
                self.env.model.geom_friction[idx, 0] = new_friction_
