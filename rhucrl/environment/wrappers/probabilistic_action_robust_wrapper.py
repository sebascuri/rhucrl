"""Base Class of an Adversarial Environments."""
from typing import Tuple

import numpy as np
from gym import Env
from gym.spaces import Box

from .adversarial_wrapper import AdversarialWrapper


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

    def __init__(self, env: Env, alpha: float) -> None:
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1] and {alpha} was given.")
        if not isinstance(env.action_space, Box):
            raise TypeError("Only continuous actions allowed.")
        super().__init__(
            env,
            adversarial_low=env.action_space.low * (0 if alpha == 0 else 1 / alpha),
            adversarial_high=env.action_space.high * (0 if alpha == 0 else 1 / alpha),
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

        # Choose action at random.
        if np.random.rand() < self.alpha:
            action = protagonist_action
        else:
            action = adversarial_action

        return self.env.step(action)
