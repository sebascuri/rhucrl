"""Python Script Template."""
from abc import ABCMeta
from typing import Optional

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Observation
from torch import Tensor


class AdversarialAgent(AbstractAgent, metaclass=ABCMeta):
    """Adversarial Agent."""

    def __init__(
        self,
        protagonist_agent,
        adversarial_agent,
        train_frequency=1,
        num_rollouts=0,
        exploration_steps=0,
        exploration_episodes=0,
        gamma=0.99,
        comment="",
    ):
        super().__init__(
            train_frequency=train_frequency,
            num_rollouts=num_rollouts,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            gamma=gamma,
            comment=comment,
        )
        protagonist_agent.comment = "Protagonist"
        adversarial_agent.comment = "Adversary"

        self.dist_params = protagonist_agent.dist_params
        self.protagonist_agent = protagonist_agent
        self.adversarial_agent = adversarial_agent

        self.protagonist_agent.train_frequency = 0
        self.protagonist_agent.num_rollouts = 0
        self.adversarial_agent.train_frequency = 0
        self.adversarial_agent.num_rollouts = 0

        self.protagonist_training = True
        self.adversarial_training = True

    def send_observations(
        self, protagonist_observation: Observation, adversarial_observation: Observation
    ) -> None:
        """Send the observations to each player."""
        self.protagonist_agent.observe(protagonist_observation)
        self.adversarial_agent.observe(adversarial_observation)

        if (
            self._training  # training mode.
            and self.total_steps >= self.exploration_steps  # enough steps.
            and self.total_episodes >= self.exploration_episodes  # enough episodes.
            and self.train_frequency > 0  # train after a transition.
            and self.total_steps % self.train_frequency == 0  # correct steps.
        ):
            self.learn()

    def learn(self) -> None:
        """Learn protagonist and adversarial agents."""
        if self.protagonist_training:
            self.protagonist_agent.learn()
        if self.adversarial_training:
            self.adversarial_agent.learn()

    def __str__(self) -> str:
        """Generate string to parse the agent."""
        str_ = super().__str__()
        str_ += str(self.protagonist_agent)
        str_ += str(self.adversarial_agent)
        return str_

    def start_episode(self) -> None:
        """Start episode of both players."""
        super().start_episode()
        self.protagonist_agent.start_episode()
        self.adversarial_agent.start_episode()

    def end_episode(self) -> None:
        """End episode of both players."""
        self.protagonist_agent.end_episode()
        self.adversarial_agent.end_episode()

        if (
            self._training  # training mode.
            and self.total_steps >= self.exploration_steps  # enough steps.
            and self.total_episodes >= self.exploration_episodes  # enough episodes.
            and self.num_rollouts > 0  # train once the episode ends.
            and (self.total_episodes + 1) % self.num_rollouts == 0  # correct steps.
        ):  # use total_episodes + 1 because the super() is called after training.
            self.learn()
        super().end_episode()

    def end_interaction(self) -> None:
        """End interaction of both players."""
        self.protagonist_agent.end_interaction()
        self.adversarial_agent.end_interaction()
        super().end_interaction()

    def set_goal(self, goal: Optional[Tensor]) -> None:
        """Set the goal to both players."""
        self.protagonist_agent.set_goal(goal)
        self.adversarial_agent.set_goal(goal)

    def train(self, val: bool = True) -> None:
        """Set training mode.

        In eval mode, both the protagonist and the adversary learn.
        """
        self.protagonist_training = True
        self.adversarial_training = True
        self.protagonist_agent.train(val)
        self.adversarial_agent.train(val)
        super().train(val)

    def eval(self, val: bool = True) -> None:
        """Set evaluation mode.

        In eval mode, both the protagonist and the adversary do not learn.
        """
        self.protagonist_training = False
        self.adversarial_training = False
        self.protagonist_agent.eval(val)
        self.adversarial_agent.eval(val)
        super().eval(val)

    def train_adversary(self) -> None:
        """Set train-adversary mode.

        In this training mode, the protagonist is kept fixed, and the adversary learns
        to hinder the protagonist.
        """
        self.protagonist_training = False
        self.protagonist_agent.eval()

        self.adversarial_training = True
        self.adversarial_agent.train()

    def only_protagonist(self, val: bool = True) -> None:
        """Evaluate the protagonist using only the protagonist policy."""
        self.policy.only_protagonist = val

    def save(self, filename, directory=None):
        """Save agent."""
        self.protagonist_agent.save("Learner" + filename, directory=directory)
        self.adversarial_agent.save("Adversary" + filename, directory=directory)
