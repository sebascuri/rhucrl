from typing import Dict, Tuple, Iterable, Optional
import numpy as np

from .adversarial_wrapper import AdversarialWrapper
import gym

try:
    from gym.envs.mujoco import MujocoEnv
    class MujocoDomainRandomizationWrapper(AdversarialWrapper):
        mass_names: Dict[str, Tuple[int, float]]
        friction_names: Dict[str, Tuple[int, np.ndarray]]
        def __init__(
            self,
            env: MujocoEnv,
            mass_names: Optional[Iterable[str]] = ...,
            friction_names: Optional[Iterable[str]] = ...,
        ) -> None: ...
        def _antagonist_action_to_mass(self, antagonist_action: np.ndarray) -> None: ...
        def _antagonist_action_to_friction(
            self, antagonist_action: np.ndarray
        ) -> None: ...

except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    pass
