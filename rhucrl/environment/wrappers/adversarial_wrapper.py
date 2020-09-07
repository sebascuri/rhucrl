"""Base Class of an Adversarial Environments."""
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
from gym import Env, Wrapper
from gym.spaces import Box


class AdversarialWrapper(Wrapper, metaclass=ABCMeta):
    r"""An adversarial environment wrapper.

    This is an abstract wrapper that wraps a gym Env.

    It overrides the step() method.
    If the action has the same dimensions as the original environment, then the step()
    method from the original environment is called.

    If the action has other dimensions, then the adversarial_step() method is called.
    AdversarialWrapper leaves this method abstract.

    """

    antagonist_low: np.ndarray
    antagonist_high: np.ndarray

    protagonist_action_dim: Tuple[int]
    antagonist_action_dim: Tuple[int]

    def __init__(
        self,
        env: Env,
        antagonist_low: np.ndarray,
        antagonist_high: np.ndarray,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(env=env)
        if not isinstance(self.action_space, Box):
            raise TypeError("Only continuous actions allowed.")
        self.protagonist_action_dim = self.env.action_space.shape
        self.antagonist_action_dim = antagonist_high.shape

        self.antagonist_low = antagonist_low
        self.antagonist_high = antagonist_high
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        """Return robustness level."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        """Set robustness level."""
        if alpha < 0:
            raise ValueError(f"alpha must be strictly positive and {alpha} was given.")
        self._alpha = alpha

        self.action_space = Box(
            low=np.concatenate(
                (self.env.unwrapped.action_space.low, alpha * self.antagonist_low)
            ),
            high=np.concatenate(
                (self.env.unwrapped.action_space.high, alpha * self.antagonist_high)
            ),
            shape=(self.protagonist_action_dim[0] + self.antagonist_action_dim[0],),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """See `gym.Env.step()'."""
        if len(action) == self.protagonist_action_dim[0]:
            assert self.env.action_space.contains(action), f"{action} invalid"
            observation, reward, done, info = self.env.step(action)
        else:
            assert self.action_space.contains(action), f"{action} invalid"

            protagonist_action = action[: self.protagonist_action_dim[0]]
            antagonist_action = action[self.protagonist_action_dim[0] :]

            observation, reward, done, info = self.adversarial_step(
                protagonist_action, antagonist_action
            )

        return observation, reward, done, info

    @abstractmethod
    def adversarial_step(
        self, protagonist_action: np.ndarray, antagonist_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """Perform an adversarial step on the environment."""
        raise NotImplementedError
