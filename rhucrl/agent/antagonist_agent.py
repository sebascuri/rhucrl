"""Python Script Template."""
from importlib import import_module

from rllib.agent.abstract_agent import AbstractAgent
from rllib.agent.random_agent import RandomAgent
from rllib.util.neural_networks.utilities import freeze_parameters

from rhucrl.algorithm.antagonist_algorithm import AntagonistAlgorithm
from rhucrl.policy.split_policy import SplitPolicy


class AntagonistAgent(AbstractAgent):
    """Antagonist Agent class."""

    @classmethod
    def default(
        cls, environment, protagonist_agent=None, base_agent_name="MVE", *args, **kwargs
    ):
        """Initialize antagonist agent."""
        if protagonist_agent is None:
            protagonist_agent = RandomAgent.default(environment, *args, **kwargs)
        try:
            protagonist_policy = protagonist_agent.policy.protagonist_policy
        except AttributeError:
            protagonist_policy = protagonist_agent.policy

        freeze_parameters(protagonist_policy)
        policy = SplitPolicy.default(environment=environment, *args, *kwargs)
        policy.set_protagonist_policy(protagonist_policy)
        policy.antagonist = True
        agent_module = import_module("rllib.agent")
        antagonist_agent = getattr(agent_module, f"{base_agent_name}Agent").default(
            environment, policy=policy, *args, **kwargs
        )
        antagonist_agent.algorithm = AntagonistAlgorithm(antagonist_agent.algorithm)
        return antagonist_agent
