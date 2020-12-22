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


class ActionRobustAgent(AbstractAgent):
    """Action Robust Agent class."""

    @classmethod
    def default(cls, environment, base_agent_name="SAC", kind="noisy", *args, **kwargs):
        """Initialize Action Robust agent."""
        if kind == "noisy":
            policy_ = NoisyActionRobustPolicy
            pathwise_loss_ = NoisyActionRobustPathwiseLoss
        elif kind == "probabilistic":
            policy_ = ProbabilisticActionRobustPolicy
            pathwise_loss_ = ProbabilisticActionRobustPathwiseLoss
        else:
            raise NotImplementedError(f"{kind} wrongly parsed.")

        policy = policy_.default(environment, *args, **kwargs)

        agent_module = import_module("rllib.agent")
        agent = getattr(agent_module, f"{base_agent_name}Agent").default(
            environment, policy=policy, *args, **kwargs
        )
        agent.algorithm.pathwise_loss = pathwise_loss_(
            critic=agent.algorithm.pathwise_loss.critic,
            policy=agent.algorithm.pathwise_loss.policy,
        )

        return agent
