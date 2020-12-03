"""Python Script Template."""
from abc import ABCMeta
from copy import deepcopy

import numpy as np
from rllib.agent import AbstractAgent
from rllib.util.logger import Logger


class AdversarialAgent(AbstractAgent, metaclass=ABCMeta):
    r"""Adversarial Agent.

    Protagonist/Weak-Antagonist optimizes
        \max_{\pi_p} \min_{\pi_a} J(\pi_p, \pi_a)

    Strong-Antagonist optimizes
        \min_{\pi_a} J(\pi_p.detach(), \pi_a)

    In RH-UCRL
     Protagonist/Weak-Antagonist optimizes
        \max_{\pi_p} \min_{\pi_a} \max_{\pi_h} J(\pi_p, \pi_a, \pi_h)

    Strong-Antagonist optimizes
        \min_{\pi_a} \min_{\pi_h} J(\pi_p.detach(), \pi_a)
    """

    def __init__(
        self,
        protagonist_agent,
        antagonist_agent,
        n_protagonists=1,
        n_antagonists=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.protagonists = [deepcopy(protagonist_agent) for _ in range(n_protagonists)]
        self.antagonists = [deepcopy(antagonist_agent) for _ in range(n_antagonists)]

        self.agents = self.protagonists + self.antagonists

        for role, agent_list in zip(
            ["Protagonist", "Antagonist"], [self.protagonists, self.antagonists]
        ):
            for i, agent in enumerate(agent_list):
                try:
                    agent.logger.delete_directory()
                    agent.logger = Logger(
                        f"{self.logger.log_dir[5:]}/{role}-{i}-{agent.name}",
                        tensorboard=False,
                    )
                except (FileNotFoundError, AttributeError):
                    pass

        self.protagonist_idx = 0
        self.antagonist_idx = 0

    @property
    def protagonist(self):
        """Get current protagonist."""
        return self.protagonists[self.protagonist_idx]

    @property
    def antagonist(self):
        """Get current antagonist."""
        return self.antagonists[self.antagonist_idx]

    def send_observations(self, protagonist_observation, antagonist_observation):
        """Send the observations to each player."""
        self.protagonist.observe(protagonist_observation)
        self.antagonist.observe(antagonist_observation)

    def __str__(self):
        """Generate string to parse the agent."""
        str_ = super().__str__()
        for agent in self.agents:
            str_ += str(agent)
        return str_

    def start_episode(self):
        """Start episode of both players."""
        super().start_episode()
        self.protagonist_idx = np.random.choice(len(self.protagonists))
        self.antagonist_idx = np.random.choice(len(self.antagonists))

        self.protagonist.start_episode()
        self.antagonist.start_episode()

    def end_episode(self):
        """End episode of both players."""
        self.protagonist.end_episode()
        self.antagonist.end_episode()
        super().end_episode()

    def end_interaction(self):
        """End interaction of both players."""
        self.protagonist.end_interaction()
        self.antagonist.end_interaction()
        super().end_interaction()

    def set_goal(self, goal):
        """Set the goal to both players."""
        for agent in self.agents:
            agent.set_goal(goal)

    def train(self, val=True):
        """Set training mode.

        In eval mode, both the protagonist and the antagonist learn.
        """
        self.protagonist.train(val=val)
        self.antagonist.train(val=val)
        super().train(val)

    def eval(self, val=True):
        """Set evaluation mode.

        In eval mode, both the protagonist and the antagonist do not learn.
        """
        self.protagonist.eval(val=val)
        self.antagonist.eval(val=val)
        super().eval(val)

    def train_only_antagonist(self):
        """Set into train antagonist mode.

        In this training mode, the protagonist is kept fixed, and the antagonist learns
        to hinder the protagonist.
        """
        self.protagonist.eval()
        self.antagonist.train()

    def train_only_protagonist(self):
        """Set into train protagonist mode.

        In this training mode, the antagonist is kept fixed, and the protagonist learns
        to hinder the antagonist.
        """
        self.protagonist.train()
        self.antagonist.eval()

    def only_protagonist(self, val=True):
        """Evaluate the protagonist using only the protagonist policy."""
        self.policy.only_protagonist = val

    def save(self, filename, directory=None):
        """Save both agents."""
        for role, agent_list in zip(
            ["Protagonist", "Antagonist"], [self.protagonists, self.antagonists]
        ):
            for i, agent in enumerate(agent_list):
                agent.save(f"{role}-{i}-{filename}", directory=directory)

    def load_protagonist(self, path, idx=0):
        """Load protagonist agent from path."""
        self.protagonists[idx].load(path)

    def load_antagonist(self, path, idx=0):
        """Load antagonist agent from path."""
        self.antagonists[idx].load(path)
