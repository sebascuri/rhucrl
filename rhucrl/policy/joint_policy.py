"""Python Script Template."""

import torch
from rllib.dataset.datatypes import State, TupleDistribution
from rllib.policy import AbstractPolicy, NNPolicy

from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.environment.utilities import (
    adversarial_to_antagonist_environment,
    adversarial_to_protagonist_environment,
)

from .adversarial_policy import AdversarialPolicy


class JointPolicy(AdversarialPolicy):
    """Given a protagonist and an antagonist policy, combine to give a joint policy."""

    protagonist_policy: AbstractPolicy
    antagonist_policy: AbstractPolicy

    def __init__(
        self, protagonist_policy: AbstractPolicy, antagonist_policy: AbstractPolicy
    ) -> None:
        super().__init__(
            dim_state=protagonist_policy.dim_state,
            protagonist_dim_action=protagonist_policy.dim_action,
            antagonist_dim_action=antagonist_policy.dim_action,
            deterministic=protagonist_policy.deterministic,
            action_scale=torch.cat(
                (protagonist_policy.action_scale, antagonist_policy.action_scale)
            ),
            dist_params=protagonist_policy.dist_params,
        )
        self.protagonist_policy = protagonist_policy
        self.antagonist_policy = antagonist_policy

    def forward(self, state: State) -> TupleDistribution:
        """Forward compute the policy."""
        protagonist_mean, protagonist_chol = self.protagonist_policy(state)
        antagonist_mean, antagonist_chol = self.antagonist_policy(state)

        return self.stack_distributions(
            protagonist_mean, protagonist_chol, antagonist_mean, antagonist_chol
        )

    @classmethod
    def default(cls, environment: AdversarialEnv, *args, **kwargs):
        """Get default policy."""
        protagonist_env = adversarial_to_protagonist_environment(environment)
        protagonist_policy = NNPolicy.default(protagonist_env, *args, **kwargs)

        antagonist_env = adversarial_to_antagonist_environment(environment)
        antagonist_policy = NNPolicy.default(antagonist_env, *args, **kwargs)

        return cls(protagonist_policy, antagonist_policy)
