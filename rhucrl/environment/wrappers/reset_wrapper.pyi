"""Reset Wrapper."""
from typing import Any, Callable

from rllib.dataset.datatypes import State
from gym import Env, Wrapper

class ResetWrapper(Wrapper):
    """Wrap environment by changing the reset function."""

    reset_function: Callable[[Wrapper, Any], State]
    def __init__(
        self, env: Env, reset_function: Callable[[Wrapper, Any], State]
    ) -> None: ...
    def reset(self, **kwargs: Any) -> State: ...
