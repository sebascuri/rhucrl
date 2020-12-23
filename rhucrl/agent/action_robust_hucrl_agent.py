"""Python Script Template."""
from importlib import import_module

import torch
from rllib.agent import ModelBasedAgent
from rllib.util.neural_networks.utilities import deep_copy_module

from rhucrl.algorithm.action_robust import (
    NoisyActionRobustPathwiseLoss,
    ProbabilisticActionRobustPathwiseLoss,
)
from rhucrl.policy.action_robust_policy import (
    NoisyActionRobustPolicy,
    ProbabilisticActionRobustPolicy,
)


class ActionRobustHUCRLAgent(ModelBasedAgent):
    """Action-Robust HUCRL Agent."""

    def __init__(self, base_agent, *args, **kwargs):
        super().__init__(
            **{**base_agent.__dict__, **dict(base_agent.algorithm.named_modules())}
        )
        self.algorithm = base_agent.algorithm
        self.antagonist_algorithm = deep_copy_module(base_agent.algorithm)

        pessimistic_policy = deep_copy_module(base_agent.policy)
        pessimistic_policy.hallucinate_antagonist = True
        pessimistic_policy.antagonist = True
        pessimistic_policy.set_protagonist_policy(
            self.algorithm.policy.protagonist_policy
        )
        self.antagonist_algorithm.set_policy(new_policy=pessimistic_policy)
        self.policy = self.antagonist_algorithm.policy

        assert self.policy is self.antagonist_algorithm.policy
        assert self.policy is not self.algorithm.policy
        assert (
            self.policy.protagonist_policy is self.algorithm.policy.protagonist_policy
        )

        # Update optimizer.
        self.optimizer = type(self.optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if ("model" not in name and "target" not in name and p.requires_grad)
            ]
            + [
                p
                for name, p in self.antagonist_algorithm.named_parameters()
                if (
                    "protagonist" not in name
                    and "model" not in name
                    and "target" not in name
                    and p.requires_grad
                )
            ],
            **self.optimizer.defaults,
        )

    def learn(self):
        """Learn antagonist."""
        # Learn Protagonist.
        super().learn()
        # # Set protagonist policy parameters.
        # self.antagonist_algorithm.policy.set_protagonist_policy(
        #     self.algorithm.policy.protagonist_policy
        # )
        # Learn Antagonist.
        self.learn_antagonist()

    def learn_antagonist(self, memory=None):
        """Fit the antagonist algorithm."""
        #

        def closure():
            """Gradient calculation."""
            if memory is None:
                observation, *_ = self.memory.sample_batch(self.batch_size)
            else:
                observation, *_ = memory.sample_batch(self.batch_size)
            self.optimizer.zero_grad()
            losses = self.antagonist_algorithm(observation.clone())
            losses.combined_loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.clip_gradient_val
            )
            return losses

        self._learn_steps(closure)

    @classmethod
    def default(
        cls, environment, base_agent_name="BPTT", kind="noisy", *args, **kwargs
    ):
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
        return super().default(
            environment=environment, base_agent=agent, *args, **kwargs
        )
