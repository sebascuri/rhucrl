"""Base Class of an Adversarial Environments."""
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Tuple

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

    adversarial_low: np.ndarray
    adversarial_high: np.ndarray

    protagonist_action_dim: Tuple[int]
    adversarial_action_dim: Tuple[int]

    def __init__(
        self,
        env: Env,
        adversarial_low: np.ndarray,
        adversarial_high: np.ndarray,
        alpha: float = 1.0,
        reward_function: Optional[Callable] = None,
    ) -> None:
        super().__init__(env=env)
        if not isinstance(self.action_space, Box):
            raise TypeError("Only continuous actions allowed.")
        self.protagonist_action_dim = self.env.action_space.shape
        self.adversarial_action_dim = adversarial_high.shape

        self.adversarial_low = adversarial_low
        self.adversarial_high = adversarial_high
        self.alpha = alpha

        self.reward_function = reward_function

    @property
    def alpha(self) -> float:
        """Return robustness level."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        """Set robustness level."""
        if alpha < 0:
            raise ValueError(f"alpha must be positive and {alpha} was given.")
        self._alpha = alpha

        self.action_space = Box(
            low=np.concatenate(
                (self.env.unwrapped.action_space.low, alpha * self.adversarial_low)
            ),
            high=np.concatenate(
                (self.env.unwrapped.action_space.high, alpha * self.adversarial_high)
            ),
            dtype=np.float32,
        )

    def reward(self, observation, action, reward, done, info):
        """Override reward calculation."""
        if self.reward_function is None:
            return reward
        else:
            return self.reward_function(observation, action, reward, done, info)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """See `gym.Env.step()'."""
        if len(action) == self.protagonist_action_dim[0]:
            assert self.env.action_space.contains(action), f"{action} invalid"
            observation, reward, done, info = self.env.step(action)
        else:
            assert self.action_space.contains(action), f"{action} invalid"

            protagonist_action = action[: self.protagonist_action_dim[0]]
            adv_action = action[self.protagonist_action_dim[0] :]

            observation, reward, done, info = self.adversarial_step(
                protagonist_action, adv_action
            )

        return (
            observation,
            self.reward(observation, action, reward, done, info),
            done,
            info,
        )

    @abstractmethod
    def adversarial_step(
        self, protagonist_action: np.ndarray, adversarial_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """Perform an adversarial step on the environment."""
        raise NotImplementedError


class ProbabilisticActionRobustWrapper(AdversarialWrapper):
    r"""Class that wraps an environment as a probabilistic action robust environment.

    A noisy action robust environments executes an action given by:

    ..math:: a = a_p, with probability = (1-\alpha)
    ..math:: a = a_a, with probability = \alpha
    where, a_p is the player's action and a_a is the adversarial action.

    Parameters
    ----------
    env: Env.
        Environment to wrap.
    alpha: float.
        Proportion of action robustness.

    Notes
    -----
    Only Continuous actions allowed.

    References
    ----------
    Tessler, C., Efroni, Y., & Mannor, S. (2019, May).
    Action Robust Reinforcement Learning and Applications in Continuous Control. ICML.
    """

    def __init__(self, env, alpha: float = 0):
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1] and {alpha} was given.")
        if not isinstance(env.action_space, Box):
            raise TypeError("Only continuous actions allowed.")
        super().__init__(
            env,
            adversarial_low=env.action_space.low,
            adversarial_high=env.action_space.high,
            alpha=alpha,
        )

    def adversarial_step(
        self, protagonist_action, adversarial_action
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """See `gym.Env.step()'."""
        assert (
            len(protagonist_action) == self.protagonist_action_dim[0]
        ), "Protagonist action has wrong dimensions."
        assert (
            len(adversarial_action) == self.protagonist_action_dim[0]
        ), "Adversarial action has wrong dimensions."
        if np.random.rand() < self.alpha:
            action = protagonist_action
        else:
            action = adversarial_action

        return self.env.step(action)


class NoisyActionRobustWrapper(AdversarialWrapper):
    r"""Class that wraps an environment as a noisy action robust environment.

    A noisy action robust environments executes an action given by:

    ..math:: a = (1 - \alpha) a_p + \alpha a_a,
    where, a_p is the player's action and a_a is the adversarial action.

    Parameters
    ----------
    env: Env.
        Environment to wrap.
    alpha: float.
        Proportion of action robustness.

    Notes
    -----
    Only Continuous actions allowed.

    References
    ----------
    Tessler, C., Efroni, Y., & Mannor, S. (2019, May).
    Action Robust Reinforcement Learning and Applications in Continuous Control. ICML.
    """

    def __init__(self, env: Env, alpha: float = 0):
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1] and {alpha} was given.")
        if not isinstance(env.action_space, Box):
            raise TypeError("Only continuous actions allowed.")
        super().__init__(
            env,
            adversarial_low=env.action_space.low,
            adversarial_high=env.action_space.high,
            alpha=alpha,
        )

    def adversarial_step(
        self, protagonist_action, adversarial_action
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """See `gym.Env.step()'."""
        assert (
            len(protagonist_action) == self.protagonist_action_dim[0]
        ), "Protagonist action has wrong dimensions."
        assert (
            len(adversarial_action) == self.protagonist_action_dim[0]
        ), "Adversarial action has wrong dimensions."
        action = (1 - self.alpha) * protagonist_action + self.alpha * adversarial_action
        return self.env.step(action)
