"""Python Script Template."""
from importlib import import_module

from rllib.agent.abstract_agent import AbstractAgent

from rhucrl.algorithm.action_robust import (
    NoisyActionRobustPathwiseLoss,
    ProbabilisticActionRobustPathwiseLoss,
)
from rhucrl.policy.action_robust_policy import (
    NoisyActionRobustPolicy,
    ProbabilisticActionRobustPolicy,
)


class NoisyActionRobustAgent(AbstractAgent):
    """Noisy Action Robust Agent class."""

    @classmethod
    def default(cls, environment, base_agent_name="SAC", *args, **kwargs):
        """Initialize antagonist agent."""
        policy = NoisyActionRobustPolicy.default(
            environment=environment, *args, *kwargs
        )

        agent_module = import_module("rllib.agent")
        agent = getattr(agent_module, f"{base_agent_name}Agent").default(
            environment, policy=policy, *args, **kwargs
        )
        agent.algorithm.pathwise_loss = NoisyActionRobustPathwiseLoss(
            critic=agent.algorithm.pathwise_loss.critic,
            policy=agent.algorithm.pathwise_loss.policy,
        )

        return agent


class ProbabilisticActionRobustAgent(AbstractAgent):
    """Probabilistic Action Robust Agent class."""

    @classmethod
    def default(cls, environment, base_agent_name="SAC", *args, **kwargs):
        """Initialize antagonist agent."""
        policy = ProbabilisticActionRobustPolicy.default(
            environment=environment, *args, *kwargs
        )

        agent_module = import_module("rllib.agent")
        agent = getattr(agent_module, f"{base_agent_name}Agent").default(
            environment, policy=policy, *args, **kwargs
        )
        agent.algorithm.pathwise_loss = ProbabilisticActionRobustPathwiseLoss(
            critic=agent.algorithm.pathwise_loss.critic,
            policy=agent.algorithm.pathwise_loss.policy,
        )

        return agent
