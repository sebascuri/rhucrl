"""Python Script Template."""
from abc import ABCMeta

from rllib.agent import AbstractAgent
from rllib.util.logger import Logger


class AdversarialAgent(AbstractAgent, metaclass=ABCMeta):
    r"""Adversarial Agent.

    Protagonist/Weak-Antagonist optimizes
        \max_{\pi_p} \min_{\pi_a} J(\pi_p, \pi_a)

    Strong-Antagonist optimizes
        \min_{\pi_a} J(\pi_p.detach(), \pi_a)

    In most cases Weak-Antagonist=Strong-Antagonist, except in RH-UCRL. In RH-UCRL

     Protagonist/Weak-Antagonist optimizes
        \max_{\pi_p} \min_{\pi_a} \max_{\pi_h} J(\pi_p, \pi_a, \pi_h)

    Strong-Antagonist optimizes
        \min_{\pi_a} \min_{\pi_h} J(\pi_p.detach(), \pi_a)
    """

    def __init__(
        self,
        protagonist_agent,
        antagonist_agent,
        weak_antagonist_agent=None,
        tensorboard=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.protagonist_agent = protagonist_agent
        self.antagonist_agent = antagonist_agent
        self.protagonist_agent.logger.delete_directory()
        self.protagonist_agent.logger = Logger(
            f"{self.logger.writer.logdir[5:]}/Protagonist", tensorboard=tensorboard
        )
        self.antagonist_agent.logger.delete_directory()
        self.antagonist_agent.logger = Logger(
            f"{self.logger.writer.logdir[5:]}/Antagonist", tensorboard=tensorboard
        )

        self.agents = [self.protagonist_agent, self.antagonist_agent]
        self.antagonist_agents = [self.antagonist_agent]
        self.weak_antagonist_agent = weak_antagonist_agent
        if weak_antagonist_agent is not None:
            self.weak_antagonist_agent.logger.delete_directory()
            self.weak_antagonist_agent.logger = Logger(
                f"{self.logger.writer.logdir[5:]}/WeakAntagonist",
                tensorboard=tensorboard,
            )
            self.agents.append(self.weak_antagonist_agent)
            self.antagonist_agents.append(self.weak_antagonist_agent)

    def send_observations(self, protagonist_observation, antagonist_observation):
        """Send the observations to each player."""
        self.protagonist_agent.observe(protagonist_observation)
        for agent in self.antagonist_agents:
            agent.observe(antagonist_observation)

    def __str__(self):
        """Generate string to parse the agent."""
        str_ = super().__str__()
        for agent in self.agents:
            str_ += str(agent)
        return str_

    def start_episode(self):
        """Start episode of both players."""
        super().start_episode()
        for agent in self.agents:
            agent.start_episode()

    def end_episode(self):
        """End episode of both players."""
        for agent in self.agents:
            agent.end_episode()
        super().end_episode()

    def end_interaction(self):
        """End interaction of both players."""
        for agent in self.agents:
            agent.end_interaction()
        super().end_interaction()

    def set_goal(self, goal):
        """Set the goal to both players."""
        for agent in self.agents:
            agent.set_goal(goal)

    def train(self, val=True):
        """Set training mode.

        In eval mode, both the protagonist and the antagonist learn.
        """
        for agent in self.agents:
            agent.train(val)
        super().train(val)

    def eval(self, val=True):
        """Set evaluation mode.

        In eval mode, both the protagonist and the antagonist do not learn.
        """
        for agent in self.agents:
            agent.eval(val)
        super().eval(val)

    def train_only_antagonist(self):
        """Set into train antagonist mode.

        In this training mode, the protagonist is kept fixed, and the antagonist learns
        to hinder the protagonist.
        """
        self.protagonist_agent.eval()
        for agent in self.antagonist_agents:
            agent.train()

    def train_only_protagonist(self):
        """Set into train protagonist mode.

        In this training mode, the antagonist is kept fixed, and the protagonist learns
        to hinder the antagonist.
        """
        self.protagonist_agent.train()
        for agent in self.antagonist_agents:
            agent.eval()

    def only_protagonist(self, val=True):
        """Evaluate the protagonist using only the protagonist policy."""
        self.policy.only_protagonist = val

    def save(self, filename, directory=None):
        """Save both agents."""
        self.protagonist_agent.save("Protagonist" + filename, directory=directory)
        self.antagonist_agent.save("Antagonist" + filename, directory=directory)
        if self.weak_antagonist_agent is not None:
            self.weak_antagonist_agent.save(
                "Antagonist" + filename, directory=directory
            )

    def load_protagonist(self, path):
        """Load protagonist agent from path."""
        self.protagonist_agent.load(path)

    def load_antagonist(self, path):
        """Load antagonist agent from path."""
        self.antagonist_agent.load(path)

    def load_weak_antagonist(self, path):
        """Load weak antagonist agent from path."""
        self.weak_antagonist_agent.load(path)
