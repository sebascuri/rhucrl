"""Python Script Template."""
from rllib.agent import AbstractAgent
from importlib import import_module
import torch
import numpy as np
from torch.distributions.uniform import Uniform


class DomainRandomizationAgent(AbstractAgent):
    def __init__(self, base_agent, num_params, *args, **kwargs):
        super().__init__(
            **{**base_agent.__dict__, **dict(base_agent.algorithm.named_modules())}
        )
        self.algorithm = base_agent.algorithm
        self.set_policy(base_agent.policy)
        self.domain_distribution = Uniform(
            low=-torch.ones(num_params), high=torch.ones(num_params)
        )
        self.domain = self.domain_distribution.sample().detach().numpy()
        self.num_params = num_params

    def act(self, state):
        action = super().act(state)
        return np.concatenate([action, self.domain])

    def start_episode(self):
        """Sample a new domain."""
        super().start_episode()
        self.domain = self.domain_distribution.sample().detach().numpy()
        self.logger.update(**{f"domain-{i}": val for i, val in enumerate(self.domain)})

    def observe(self, observation):
        observation.action = observation.action[: self.policy.dim_action[0]]
        super().observe(observation)

    @classmethod
    def default(
        cls, environment, base_agent_name="SAC", num_params=None, *args, **kwargs
    ):
        if num_params is None:
            num_params = environment.antagonist_dim_action[0]

        agent_module = import_module("rllib.agent")
        base_agent = getattr(agent_module, f"{base_agent_name}Agent").default(
            environment, *args, **kwargs
        )
        return super().default(
            environment=environment,
            base_agent=base_agent,
            num_params=num_params,
            *args,
            **kwargs,
        )
