"""Reward Wrapper."""
from typing import Callable, Tuple

import numpy as np
from gym import Env, Wrapper

class RewardWrapper(Wrapper):
    reward_function: Callable[[np.ndarray, np.ndarray, np.ndarray, dict], float]
    def __init__(
        self,
        env: Env,
        reward_function: Callable[[np.ndarray, np.ndarray, np.ndarray, dict], float],
    ) -> None: ...
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]: ...
