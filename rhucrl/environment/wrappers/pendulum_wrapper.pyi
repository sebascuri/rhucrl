"""Adversarial Pendulum."""

from typing import List, Tuple

import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv

from .adversarial_wrapper import AdversarialWrapper

class OtherPendulum(PendulumEnv):
    """Other Pendulum overrides step method of pendulum."""

    state: np.ndarray
    def step(self, u: np.array) -> Tuple[np.ndarray, float, bool, dict]: ...

class PendulumAdvEnv(AdversarialWrapper):
    """Adversarial Pendulum Environment."""

    env: OtherPendulum
    attacks: List[str] = ...
    attack_mode: str
    def __init__(self, alpha: float = ..., attack_mode: str = ...) -> None: ...
    def adversarial_step(
        self, protagonist_action: np.ndarray, antagonist_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, dict]: ...