"""Reset Wrapper."""
from gym import Wrapper


class ResetWrapper(Wrapper):
    """Wrap environment by changing the reset function."""

    def __init__(self, env, reset_function) -> None:
        super().__init__(env)
        self.reset_function = reset_function

    def reset(self, **kwargs):
        """Wrap reward function."""
        return self.reset_function(self, **kwargs)
