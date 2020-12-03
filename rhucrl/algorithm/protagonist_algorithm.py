"""Python Script Template."""
from rllib.algorithms.derived_algorithm import DerivedAlgorithm

from rhucrl.policy.adversarial_policy import AdversarialPolicy
from rhucrl.policy.utilities import ProtagonistMode


class ProtagonistAlgorithm(DerivedAlgorithm):
    """A protagonist algorithm.

    It optimizes the loss for the base algorithm using the protagonist mode.
    """

    policy: AdversarialPolicy

    def forward(self, observation, *args, **kwargs):
        """Compute the losses.

        Given an Observation, it will compute the losses.
        Given a list of Trajectories, it tries to stack them to vectorize operations.
        If it fails, will iterate over the trajectories.
        """
        with ProtagonistMode(self.policy):
            loss = self.base_algorithm(observation)

        return loss
