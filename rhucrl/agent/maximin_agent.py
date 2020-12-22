"""Python Script Template."""
from importlib import import_module

from rllib.agent import AbstractAgent

from rhucrl.algorithm.maximin_algorithm import MaxiMinAlgorithm
from rhucrl.policy.split_policy import SplitPolicy


class MaxiMinAgent(AbstractAgent):
    """Maximin Agent class."""

    @classmethod
    def default(cls, environment, base_agent_name="BPTT", *args, **kwargs):
        """Initialize maximin by default."""
        policy = SplitPolicy.default(
            environment, hallucinate_protagonist=True, *args, **kwargs
        )
        agent_module = import_module("rllib.agent")
        base_agent = getattr(agent_module, f"{base_agent_name}Agent").default(
            environment, policy=policy, *args, **kwargs
        )
        base_agent.algorithm = MaxiMinAlgorithm(base_agent.algorithm)
        return base_agent
