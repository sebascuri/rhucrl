"""Python Script Template."""
from abc import ABCMeta
from typing import Optional

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Observation
from torch import Tensor

from rhucrl.policy.joint_policy import JointPolicy


class AdversarialAgent(AbstractAgent, metaclass=ABCMeta):
    """Adversarial Agent."""

    def __init__(self, protagonist_agent, antagonist_agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        protagonist_agent.comment = "Protagonist"
        antagonist_agent.comment = "Antagonist"

        self.protagonist_agent = protagonist_agent
        self.antagonist_agent = antagonist_agent

        self.protagonist_agent.train_frequency = 0
        self.protagonist_agent.num_rollouts = 0
        self.antagonist_agent.train_frequency = 0
        self.antagonist_agent.num_rollouts = 0

        self.protagonist_training = True
        self.antagonist_training = True

        self.policy = JointPolicy(
            self.protagonist_agent.policy, self.antagonist_agent.policy
        )

    def send_observations(
        self, protagonist_observation: Observation, antagonist_observation: Observation
    ) -> None:
        """Send the observations to each player."""
        self.protagonist_agent.observe(protagonist_observation)
        self.antagonist_agent.observe(antagonist_observation)

        if self.train_at_observe:
            self.learn()

    def learn(self) -> None:
        """Learn protagonist and antagonist agents."""
        if self.protagonist_training:
            self.protagonist_agent.learn()
        if self.antagonist_training:
            self.antagonist_agent.learn()

    def __str__(self) -> str:
        """Generate string to parse the agent."""
        str_ = super().__str__()
        str_ += str(self.protagonist_agent)
        str_ += str(self.antagonist_agent)
        return str_

    def start_episode(self) -> None:
        """Start episode of both players."""
        super().start_episode()
        self.protagonist_agent.start_episode()
        self.antagonist_agent.start_episode()

    def end_episode(self) -> None:
        """End episode of both players."""
        self.protagonist_agent.end_episode()
        self.antagonist_agent.end_episode()

        if self.train_at_end_episode:
            self.learn()
        super().end_episode()

    def end_interaction(self) -> None:
        """End interaction of both players."""
        self.protagonist_agent.end_interaction()
        self.antagonist_agent.end_interaction()
        super().end_interaction()

    def set_goal(self, goal: Optional[Tensor]) -> None:
        """Set the goal to both players."""
        self.protagonist_agent.set_goal(goal)
        self.antagonist_agent.set_goal(goal)

    def train(self, val: bool = True) -> None:
        """Set training mode.

        In eval mode, both the protagonist and the antagonist learn.
        """
        self.protagonist_training = True
        self.antagonist_training = True
        self.protagonist_agent.train(val)
        self.antagonist_agent.train(val)
        super().train(val)

    def eval(self, val: bool = True) -> None:
        """Set evaluation mode.

        In eval mode, both the protagonist and the antagonist do not learn.
        """
        self.protagonist_training = False
        self.antagonist_training = False
        self.protagonist_agent.eval(val)
        self.antagonist_agent.eval(val)
        super().eval(val)

    def train_antagonist(self) -> None:
        """Set train-antagonist mode.

        In this training mode, the protagonist is kept fixed, and the antagonist learns
        to hinder the protagonist.
        """
        self.protagonist_training = False
        self.protagonist_agent.eval()

        self.antagonist_training = True
        self.antagonist_agent.train()

    def only_protagonist(self, val: bool = True) -> None:
        """Evaluate the protagonist using only the protagonist policy."""
        self.policy.only_protagonist = val

    def save(self, filename, directory=None):
        """Save agent."""
        self.protagonist_agent.save("Protagonist" + filename, directory=directory)
        self.antagonist_agent.save("Antagonist" + filename, directory=directory)
