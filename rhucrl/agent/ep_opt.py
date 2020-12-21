"""Python Script Template."""
from rllib.agent import PPOAgent
import math
import torch
import numpy as np
from torch.distributions.uniform import Uniform
from rllib.dataset.utilities import stack_list_of_tuples


class EPOPTAgent(PPOAgent):
    def __init__(self, num_params, alpha=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_distribution = Uniform(
            low=-torch.ones(num_params), high=torch.ones(num_params)
        )
        self.domain = self.domain_distribution.sample().detach().numpy()
        self.num_params = num_params
        self.alpha = alpha

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

    def learn(self):
        """Train Policy Gradient Agent."""
        all_trajectories = [stack_list_of_tuples(t).clone() for t in self.trajectories]
        sorted_trajectories = [
            t for t in sorted(all_trajectories, key=lambda x: x.reward.sum())
        ]
        num_trajectories = math.ceil(self.alpha * len(self.trajectories))
        trajectories = sorted_trajectories[:num_trajectories]

        def closure():
            """Gradient calculation."""
            self.optimizer.zero_grad()
            losses = self.algorithm(trajectories)
            losses.combined_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.clip_gradient_val
            )

            return losses

        self._learn_steps(closure)

    @classmethod
    def default(cls, environment, num_params=None, *args, **kwargs):
        if num_params is None:
            num_params = environment.antagonist_dim_action[0]
        return super().default(environment, num_params=num_params, *args, **kwargs)
