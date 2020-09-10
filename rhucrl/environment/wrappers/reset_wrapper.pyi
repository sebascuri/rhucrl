"""Reset Wrapper."""
from typing import Any, Callable

from gym import Env, Wrapper
from rllib.dataset.datatypes import State

class ResetWrapper(Wrapper):
    """Wrap environment by changing the reset function."""

    reset_function: Callable[[Wrapper, Any], State]
    def __init__(
        self, env: Env, reset_function: Callable[[Wrapper, Any], State]
    ) -> None: ...
    def reset(self, **kwargs: Any) -> State: ...
