"""Base Class of an Adversarial Environments."""
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
from gym import Env, Wrapper

class AdversarialWrapper(Wrapper, metaclass=ABCMeta):
    antagonist_low: np.ndarray
    antagonist_high: np.ndarray
    protagonist_dim_action: Tuple[int]
    antagonist_dim_action: Tuple[int]
    def __init__(
        self,
        env: Env,
        antagonist_low: np.ndarray,
        antagonist_high: np.ndarray,
        alpha: float = 1.0,
    ) -> None: ...
    @property
    def alpha(self) -> float: ...
    @alpha.setter
    def alpha(self, alpha: float) -> None: ...
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]: ...
    @abstractmethod
    def adversarial_step(
        self, protagonist_action: np.ndarray, antagonist_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, dict]: ...
