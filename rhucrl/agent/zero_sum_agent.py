"""Zero sum agent."""
from importlib import import_module

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Observation

from rhucrl.policy.split_policy import SplitPolicy

from .adversarial_agent import AdversarialAgent


class ZeroSumAgent(AdversarialAgent):
    """Zero-Sum Agent.

    Zero-Sum has two dependent agents.
    The protagonist receives (s, a, r, s') and the protagonist (s, a, -r, s').
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.protagonist_agent.policy.base_policy
            is self.antagonist_agent.policy.base_policy
        ), "Protagonist and Adversarial agent should share the base policy."
        self.policy = self.antagonist_agent.policy

    def observe(self, observation):
        """Send observations to both players.

        The protagonist receives (s, a, r, s') and the antagonist (s, a, -r, s').

        """
        super().observe(observation)
        protagonist_observation = Observation(*observation)
        antagonist_observation = Observation(*observation)
        antagonist_observation.reward = -observation.reward

        self.send_observations(protagonist_observation, antagonist_observation)

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """Get default Zero-Sum agent."""
        p_agent, a_agent = ZeroSumAgent.get_default_agents(environment, *args, **kwargs)

        return super().default(
            environment,
            protagonist_agent=p_agent,
            antagonist_agent=a_agent,
            *args,
            **kwargs,
        )

    @staticmethod
    def get_default_protagonist(environment, protagonist_name="SAC", *args, **kwargs):
        """Get protagonist using RARL."""
        agent_ = getattr(import_module("rllib.agent"), f"{protagonist_name}Agent")
        protagonist_agent = agent_.default(
            environment, comment="Protagonist", *args, **kwargs
        )
        protagonist_policy = SplitPolicy(
            base_policy=protagonist_agent.policy,
            protagonist_dim_action=environment.protagonist_dim_action,
            antagonist_dim_action=environment.antagonist_dim_action,
            protagonist=True,
        )
        protagonist_agent.set_policy(protagonist_policy)
        return protagonist_agent

    @staticmethod
    def get_default_antagonist(
        environment,
        base_policy,
        strong_antagonist=True,
        antagonist_name="SAC",
        *args,
        **kwargs,
    ) -> AbstractAgent:
        """Get protagonist using RARL."""
        antagonist_agent = getattr(
            import_module("rllib.agent"), f"{antagonist_name}Agent"
        ).default(
            environment,
            comment=f"{'Strong' if strong_antagonist else 'Weak'} Antagonist",
            *args,
            **kwargs,
        )
        antagonist_policy = SplitPolicy(
            base_policy=base_policy,
            protagonist_dim_action=environment.protagonist_dim_action,
            antagonist_dim_action=environment.antagonist_dim_action,
            protagonist=False,
            weak_antagonist=not strong_antagonist,
            strong_antagonist=strong_antagonist,
        )
        antagonist_agent.set_policy(antagonist_policy)
        return antagonist_agent

    @staticmethod
    def get_default_agents(environment, strong_antagonist=True, *args, **kwargs):
        """Get default RARL agent."""
        p_agent = ZeroSumAgent.get_default_protagonist(environment, *args, **kwargs)
        a_agent = ZeroSumAgent.get_default_antagonist(
            environment,
            base_policy=p_agent.policy.base_policy,
            strong_antagonist=strong_antagonist,
            *args,
            **kwargs,
        )
        return p_agent, a_agent
