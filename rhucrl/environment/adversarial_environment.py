"""Base Class of an Adversarial Environments."""
from typing import Tuple

from gym.spaces import Box
from rllib.environment import GymEnvironment


class AdversarialEnv(GymEnvironment):
    """Class that wraps an adversarial environment."""

    action_space: Box

    @property
    def protagonist_dim_action(self) -> Tuple[int]:
        """Get protagonist action dimensions."""
        return self.env.unwrapped.action_space.shape

    @property
    def antagonist_dim_action(self) -> Tuple[int]:
        """Get antagonist action dimensions."""
        return (self.env.action_space.shape[0] - self.protagonist_dim_action[0],)

    @property
    def alpha(self):
        """Get robustness level."""
        try:
            return self.env.alpha
        except AttributeError:
            return 0.0

    @alpha.setter
    def alpha(self, alpha):
        """Set robustness level."""
        if not (alpha >= 0):
            raise ValueError(f"alpha must be >= 0 and {alpha} was given.")

        if hasattr(self.env, "alpha"):
            self.env.alpha = alpha
            self.action_space = self.env.action_space
