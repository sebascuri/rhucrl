"""Python Script Template."""
from importlib import import_module

import torch
from rllib.agent import ModelBasedAgent
from rllib.util.neural_networks.utilities import deep_copy_module

from rhucrl.algorithm.antagonist_algorithm import AntagonistAlgorithm
from rhucrl.algorithm.maximin_algorithm import MaxiMinAlgorithm
from rhucrl.policy.split_policy import SplitPolicy


class RHUCRLAgent(ModelBasedAgent):
    """RHUCRL Agent."""

    def __init__(self, base_agent, best_response=False, *args, **kwargs):
        super().__init__(
            **{**base_agent.__dict__, **dict(base_agent.algorithm.named_modules())}
        )
        self.algorithm = MaxiMinAlgorithm(base_algorithm=base_agent.algorithm)
        self.antagonist_algorithm = AntagonistAlgorithm(
            base_algorithm=base_agent.algorithm
        )

        pessimistic_policy = deep_copy_module(base_agent.policy)
        if best_response:
            pessimistic_policy.hallucinate_antagonist = False
        else:
            pessimistic_policy.hallucinate_antagonist = True
        pessimistic_policy.antagonist = True
        pessimistic_policy.set_protagonist_policy(
            self.algorithm.policy.protagonist_policy
        )
        self.antagonist_algorithm.set_policy(new_policy=pessimistic_policy)
        self.policy = self.antagonist_algorithm.policy

        for algorithm in [self.algorithm, self.antagonist_algorithm]:
            assert self.dynamical_model is algorithm.base_algorithm.dynamical_model
            assert self.reward_model is algorithm.base_algorithm.reward_model
            assert self.termination_model is algorithm.base_algorithm.termination_model

        assert (
            self.algorithm.base_algorithm.critic
            is self.antagonist_algorithm.base_algorithm.critic
        )
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
                if (
                    "antagonist" not in name
                    and "model" not in name
                    and "target" not in name
                    and p.requires_grad
                )
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
        # Set protagonist policy parameters.
        self.antagonist_algorithm.policy.set_protagonist_policy(
            self.algorithm.policy.protagonist_policy
        )
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
    def default(cls, environment, base_agent_name="MVE", *args, **kwargs):
        """See `AbstractAgent.default' method."""
        policy = SplitPolicy.default(
            environment, hallucinate_protagonist=True, *args, **kwargs
        )
        agent_module = import_module("rllib.agent")
        base_agent = getattr(agent_module, f"{base_agent_name}Agent").default(
            environment, policy=policy, *args, **kwargs
        )
        return super().default(
            environment=environment, base_agent=base_agent, *args, **kwargs
        )


class BestResponseAgent(RHUCRLAgent):
    """Best Response Agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(best_response=True, *args, **kwargs)
