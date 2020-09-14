"""Python Script Template."""

from typing import Any, Dict, List, Optional, Tuple

import gym.error
import numpy as np

from .adversarial_wrapper import AdversarialWrapper

try:
    from gym.envs.mujoco import MujocoEnv
    class MujocoAdversarialWrapper(AdversarialWrapper):
        """Wrapper for Mujoco adversarial environments."""

        _antagonist_body_index: List[int]
        def __init__(
            self,
            env: MujocoEnv,
            alpha: float = ...,
            force_body_names: Optional[List[str]] = ...,
            new_mass: Optional[Dict[str, float]] = ...,
            new_friction: Optional[Dict[str, float]] = ...,
        ) -> None: ...
        def _antagonist_action_to_xfrc(self, antagonist_action: np.ndarray) -> None: ...
        def adversarial_step(
            self, protagonist_action: np.ndarray, antagonist_action: np.ndarray
        ) -> Tuple[np.ndarray, float, bool, dict]: ...

except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    pass
