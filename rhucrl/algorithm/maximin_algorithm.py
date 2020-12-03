"""Python Script Template."""
from rllib.algorithms.derived_algorithm import DerivedAlgorithm

from rhucrl.policy.adversarial_policy import AdversarialPolicy
from rhucrl.policy.utilities import AntagonistMode, ProtagonistMode


class MaxiMinAlgorithm(DerivedAlgorithm):
    """A maxi-min algorithm.

    It optimizes the loss for the base algorithm using the protagonist mode.
    It then uses the -actor loss for the antagonist.
    """

    policy: AdversarialPolicy

    def set_protagonist_policy(self, new_policy):
        """Set protagonist policy."""
        self.policy.set_protagonist_policy(new_policy)
        self.base_algorithm.set_policy(self.policy)

    def set_antagonist_policy(self, new_policy):
        """Set protagonist policy."""
        self.policy.set_antagonist_policy(new_policy)
        self.base_algorithm.set_policy(self.policy)

    def forward(self, observation, *args, **kwargs):
        """Compute the losses.

        Given an Observation, it will compute the losses.
        Given a list of Trajectories, it tries to stack them to vectorize operations.
        If it fails, will iterate over the trajectories.
        """
        self.base_algorithm.reset_info()

        with ProtagonistMode(self.policy):
            protagonist_loss = self.base_algorithm(observation)
        with AntagonistMode(self.policy):
            antagonist_loss = self.base_algorithm(observation)
            antagonist_loss.policy_loss = -antagonist_loss.policy_loss

        return protagonist_loss + antagonist_loss
