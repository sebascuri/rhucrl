"""Base Class of an Adversarial Environments."""
from typing import Tuple

import numpy as np
from gym import Env
from gym.spaces import Box

from .adversarial_wrapper import AdversarialWrapper

class NoisyActionRobustWrapper(AdversarialWrapper):
    def __init__(self, env: Env, alpha: float) -> None: ...
    def adversarial_step(
        self, protagonist_action: np.ndarray, antagonist_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, dict]: ...
