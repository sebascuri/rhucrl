"""Python Script Template."""
from importlib import import_module

from .rhucrl_agent import RHUCRLAgent
from rhucrl.algorithm.action_robust import (
    NoisyActionRobustPathwiseLoss,
    ProbabilisticActionRobustPathwiseLoss,
)
from rhucrl.policy.action_robust_policy import (
    NoisyActionRobustPolicy,
    ProbabilisticActionRobustPolicy,
)


class NoisyActionRHUCRLAgent(RHUCRLAgent):
    """NoisyActionRHUCRL Agent."""

    @classmethod
    def default(cls, environment, base_agent_name="MVE", *args, **kwargs):
        """See `AbstractAgent.default' method."""
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
        return RHUCRLAgent.default(
            environment=environment, base_agent=agent, *args, **kwargs
        )


class ProbabilisticActionRHUCRLAgent(RHUCRLAgent):
    """Probabilistic-Action-RHUCRL Agent."""

    @classmethod
    def default(cls, environment, base_agent_name="MVE", *args, **kwargs):
        """See `AbstractAgent.default' method."""
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
        return RHUCRLAgent.default(
            environment=environment, base_agent=agent, *args, **kwargs
        )
