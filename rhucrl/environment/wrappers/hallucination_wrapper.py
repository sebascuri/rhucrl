"""Base Class of an Adversarial Environments."""
from typing import Tuple

import numpy as np
from gym import Env, Wrapper
from gym.spaces import Box


class HallucinationWrapper(Wrapper):
    r"""A hallucination environment wrapper.

    It overrides the step() method and it proved an protagonist_dim_action property.
    """

    def __init__(self, env: Env, protagonist: bool = True) -> None:
        super().__init__(env=env)
        self.protagonist = protagonist
        self.action_space = Box(
            low=np.concatenate(
                (self.env.action_space.low, -np.ones(self.env.observation_space.shape))
            ),
            high=np.concatenate(
                (self.env.action_space.high, +np.ones(self.env.observation_space.shape))
            ),
            shape=(self.original_action_dim[0] + self.env.observation_space.shape[0],),
            dtype=np.float32,
        )

    @property
    def original_action_dim(self) -> Tuple[int]:
        """Get original action dimension."""
        return self.env.action_space.shape

    @property
    def protagonist_dim_action(self) -> Tuple[int]:
        """Get original action dimension."""
        if self.protagonist:
            return (
                self.env.unwrapped.action_space.shape[0]
                + self.env.observation_space.shape[0],
            )
        else:
            return self.env.unwrapped.action_space.shape

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """See `gym.Env.step()'."""
        return self.env.step(action[: self.original_action_dim[0]])
