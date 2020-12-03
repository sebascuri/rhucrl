"""Python Script Template."""
from rllib.algorithms.derived_algorithm import DerivedAlgorithm
from rllib.dataset.datatypes import Loss

from rhucrl.policy.adversarial_policy import AdversarialPolicy
from rhucrl.policy.utilities import AntagonistMode


class AntagonistAlgorithm(DerivedAlgorithm):
    """A zero-sum algorithm.

    It optimizes the loss for the base algorithm using the protagonist mode.
    It then uses the -actor loss for the antagonist.
    """

    policy: AdversarialPolicy

    def forward(self, observation, *args, **kwargs):
        """Compute the losses.

        Given an Observation, it will compute the losses.
        Given a list of Trajectories, it tries to stack them to vectorize operations.
        If it fails, will iterate over the trajectories.
        """
        with AntagonistMode(self.policy):
            loss = self.base_algorithm(observation)
            loss.policy_loss = -loss.policy_loss

        return Loss(*map(lambda x: x, loss))
