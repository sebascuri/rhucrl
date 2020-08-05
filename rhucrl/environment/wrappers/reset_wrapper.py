"""Reset Wrapper."""
from typing import Any, Callable

import numpy as np
from gym import Env, Wrapper


class ResetWrapper(Wrapper):
    """Wrap environment by changing the reset function."""

    # reset_function: Callable[[Wrapper, Any], np.ndarray]

    def __init__(
        self, env: Env, reset_function: Callable[[Wrapper, Any], np.ndarray]
    ) -> None:
        super().__init__(env)
        self.reset_function = reset_function

    def reset(self, **kwargs):
        """Wrap reward function."""
        return self.reset_function(self, **kwargs)
