"""Python Script Template."""
from abc import ABCMeta
from typing import Optional

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Observation
from rllib.util.logger import Logger
from torch import Tensor


class AdversarialAgent(AbstractAgent, metaclass=ABCMeta):
    """Adversarial Agent."""

    def __init__(self, protagonist_agent, antagonist_agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protagonist_agent = protagonist_agent
        self.antagonist_agent = antagonist_agent
        self.protagonist_agent.logger.delete_directory()
        self.protagonist_agent.logger = Logger(
            f"{self.logger.writer.logdir[5:]}/Protagonist",
            tensorboard=kwargs.get("tensorboard", False),
        )
        self.antagonist_agent.logger.delete_directory()
        self.antagonist_agent.logger = Logger(
            f"{self.logger.writer.logdir[5:]}/Antagonist",
            tensorboard=kwargs.get("tensorboard", False),
        )

    def send_observations(
        self, protagonist_observation: Observation, antagonist_observation: Observation
    ) -> None:
        """Send the observations to each player."""
        self.protagonist_agent.observe(protagonist_observation)
        self.antagonist_agent.observe(antagonist_observation)

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
        self.protagonist_agent.train(val)
        self.antagonist_agent.train(val)
        super().train(val)

    def eval(self, val: bool = True) -> None:
        """Set evaluation mode.

        In eval mode, both the protagonist and the antagonist do not learn.
        """
        self.protagonist_agent.eval(val)
        self.antagonist_agent.eval(val)
        super().eval(val)

    def train_only_antagonist(self) -> None:
        """Set into train antagonist mode.

        In this training mode, the protagonist is kept fixed, and the antagonist learns
        to hinder the protagonist.
        """
        self.protagonist_agent.eval()
        self.antagonist_agent.train()

    def train_only_protagonist(self) -> None:
        """Set into train protagonist mode.

        In this training mode, the antagonist is kept fixed, and the protagonist learns
        to hinder the antagonist.
        """
        self.protagonist_agent.train()
        self.antagonist_agent.eval()

    def only_protagonist(self, val: bool = True) -> None:
        """Evaluate the protagonist using only the protagonist policy."""
        self.policy.only_protagonist = val

    def save(self, filename, directory=None):
        """Save both agents."""
        self.protagonist_agent.save("Protagonist" + filename, directory=directory)
        self.antagonist_agent.save("Antagonist" + filename, directory=directory)

    def load_protagonist(self, path):
        """Load protagonist agent from path."""
        self.protagonist_agent.load(path)

    def load_antagonist(self, path):
        """Load antagonist agent from path."""
        self.protagonist_agent.load(path)
