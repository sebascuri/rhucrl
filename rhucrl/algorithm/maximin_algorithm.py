"""Python Script Template."""
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm

from rhucrl.policy.adversarial_policy import AntagonistPolicy, ProtagonistPolicy


class MaxiMinAlgorithm(AbstractAlgorithm):
    """A maxi-min algorithm.

    It optimizes the loss for the base algorithm using the protagonist mode.
    It then uses the -actor loss for the antagonist.
    """

    def __init__(self, base_algorithm):
        super().__init__(
            **{**base_algorithm.__dict__, **dict(base_algorithm.named_modules())}
        )
        self.base_algorithm = base_algorithm
        self.base_algorithm_name = self.base_algorithm.__class__.__name__

    def update(self):
        """Update base algorithm."""
        self.base_algorithm.update()

    def reset(self):
        """Reset base algorithm."""
        self.base_algorithm.reset()

    def info(self):
        """Get info from base algorithm."""
        return self.base_algorithm.info()

    def reset_info(self):
        """Reset info from base algorithm."""
        self.base_algorithm.reset_info()

    def set_policy(self, new_policy):
        """Set policy in base algorithm."""
        self.policy = new_policy
        self.base_algorithm.set_policy(new_policy)

    def set_protagonist_policy(self, new_policy):
        """Set protagonist policy."""
        self.policy.protagonist_policy = new_policy
        self.base_algorithm.set_policy(self.policy)

    def set_antagonist_policy(self, new_policy):
        """Set protagonist policy."""
        self.policy.antagonist_policy = new_policy
        self.base_algorithm.set_policy(self.policy)

    def forward(self, observation, *args, **kwargs):
        """Compute the losses.

        Given an Observation, it will compute the losses.
        Given a list of Trajectories, it tries to stack them to vectorize operations.
        If it fails, will iterate over the trajectories.
        """
        self.base_algorithm.reset_info()

        with ProtagonistPolicy(self.policy):
            protagonist_loss = self.base_algorithm(observation)
        with AntagonistPolicy(self.policy):
            antagonist_loss = self.base_algorithm(observation)
            antagonist_loss.policy_loss = -antagonist_loss.policy_loss

        return protagonist_loss + antagonist_loss
