from typing import Any, Dict, Sequence, Tuple
import numpy as np

from rhucrl.environment import AdversarialEnv
from .adversarial_wrapper import AdversarialWrapper

class AdversarialPendulumWrapper(AdversarialWrapper):
    force_body_names: Dict[str, int]
    def __init__(
        self,
        env: AdversarialEnv,
        alpha: float = ...,
        force_body_names: Sequence[str] = ...,
        **kwargs: Any
    ) -> None: ...
    def adversarial_step(
        self, protagonist_action: np.ndarray, antagonist_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, dict]: ...
