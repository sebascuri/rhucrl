from typing import Any, Dict, Sequence, Tuple

import numpy as np

from rhucrl.environment import AdversarialEnv

from .mujoco_wrapper import MujocoAdversarialWrapper

class AdversarialPendulumWrapper(MujocoAdversarialWrapper):
    def __init__(
        self,
        env: AdversarialEnv,
        alpha: float = ...,
        force_body_names: Sequence[str] = ...,
        **kwargs: Any,
    ) -> None: ...
