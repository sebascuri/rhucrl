"""Base Class of an Adversarial Environments."""
from rllib.environment import AbstractEnvironment, GymEnvironment
from rllib.environment.utilities import parse_space


class AdversarialEnv(GymEnvironment):
    """Class that wraps an adversarial environment."""

    def pop_wrapper(self):
        """Add a wrapper for the environment."""
        self.env = self.env.env

        dim_action, num_actions = parse_space(self.env.action_space)
        dim_state, num_states = parse_space(self.env.observation_space)
        if num_states > -1:
            num_states += 1  # Add a terminal state.

        AbstractEnvironment.__init__(
            self,
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            num_observations=num_states,
        )
        self._time = 0

    @property
    def protagonist_dim_action(self):
        """Get protagonist action dimensions."""
        try:
            return self.env.protagonist_dim_action
        except AttributeError:
            return self.env.unwrapped.action_space.shape

    @property
    def antagonist_dim_action(self):
        """Get antagonist action dimensions."""
        try:
            return self.env.antagonist_dim_action
        except AttributeError:
            return (self.env.action_space.shape[0] - self.protagonist_dim_action[0],)

    @property
    def alpha(self):
        """Get robustness level."""
        try:
            return self.env.alpha
        except AttributeError:
            return 0.0

    @alpha.setter
    def alpha(self, alpha):
        """Set robustness level."""
        if not (alpha >= 0):
            raise ValueError(f"alpha must be >= 0 and {alpha} was given.")

        if hasattr(self.env, "alpha"):
            self.env.alpha = alpha
            self.action_space = self.env.action_space

    @property
    def name(self):
        """Return name."""
        wrapper = self.env.name if hasattr(self.env, "name") else ""
        return f"{self.env_name} {wrapper} {self.alpha}"
