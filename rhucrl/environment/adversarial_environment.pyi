"""Base Class of an Adversarial Environments."""
from typing import Tuple

from gym.spaces import Box
from rllib.environment import GymEnvironment

class AdversarialEnv(GymEnvironment):
    """Class that wraps an adversarial environment."""

    action_space: Box
    @property
    def protagonist_dim_action(self) -> Tuple[int]: ...
    @property
    def antagonist_dim_action(self) -> Tuple[int]: ...
    @property
    def alpha(self) -> float: ...
    @alpha.setter
    def alpha(self, alpha: float) -> None: ...
