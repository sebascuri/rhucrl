"""Python Script Template."""
from rhucrl.algorithm.maximin_algorithm import MaxiMinAlgorithm
from rhucrl.policy.adversarial_policy import ProtagonistPolicy


class ProtagonistAlgorithm(MaxiMinAlgorithm):
    """A protagonist algorithm.

    It optimizes the loss for the base algorithm using the protagonist mode.
    It then uses the -actor loss for the antagonist.
    """

    def forward(self, observation, *args, **kwargs):
        """Compute the losses.

        Given an Observation, it will compute the losses.
        Given a list of Trajectories, it tries to stack them to vectorize operations.
        If it fails, will iterate over the trajectories.
        """
        with ProtagonistPolicy(self.policy):
            loss = self.base_algorithm(observation)

        return loss
