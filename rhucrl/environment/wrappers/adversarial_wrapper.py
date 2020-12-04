"""Base Class of an Adversarial Environments."""
from abc import ABCMeta, abstractmethod

import numpy as np
from gym import Wrapper
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

    def __init__(self, env, antagonist_low, antagonist_high, alpha=1.0):
        super().__init__(env=env)
        if not isinstance(self.action_space, Box):
            raise TypeError("Only continuous actions allowed.")
        self.protagonist_dim_action = self.env.action_space.shape
        if alpha > 0:
            self.antagonist_dim_action = antagonist_high.shape
        else:
            self.antagonist_dim_action = (0,)

        self.antagonist_low = antagonist_low
        self.antagonist_high = antagonist_high
        self.alpha = alpha

    @property
    def alpha(self):
        """Return robustness level."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        """Set robustness level."""
        if alpha < 0:
            raise ValueError(f"alpha must be strictly positive and {alpha} was given.")
        self._alpha = alpha

        if alpha == 0:
            self.action_space = self.env.unwrapped.action_space
        else:
            self.action_space = Box(
                low=np.concatenate(
                    (self.env.unwrapped.action_space.low, self.antagonist_low)
                ),
                high=np.concatenate(
                    (self.env.unwrapped.action_space.high, self.antagonist_high)
                ),
                shape=(self.protagonist_dim_action[0] + self.antagonist_dim_action[0],),
                dtype=np.float32,
            )

    def step(self, action):
        """See `gym.Env.step()'."""
        if len(action) == self.protagonist_dim_action[0]:
            assert self.env.action_space.contains(action), f"{action} invalid"
            observation, reward, done, info = self.env.step(action)
        else:
            assert self.action_space.contains(action), f"{action} invalid"

            protagonist_action = action[: self.protagonist_dim_action[0]]
            antagonist_action = action[self.protagonist_dim_action[0] :]

            observation, reward, done, info = self.adversarial_step(
                protagonist_action, antagonist_action
            )

        return observation, reward, done, info

    @abstractmethod
    def adversarial_step(self, protagonist_action, antagonist_action):
        """Perform an adversarial step on the environment."""
        raise NotImplementedError

    @property
    def name(self):
        """Adversarial-Wrapper name."""
        return self.__class__.__name__
