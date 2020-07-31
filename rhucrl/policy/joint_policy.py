"""Python Script Template."""

import torch
from rllib.dataset.datatypes import State, TupleDistribution
from rllib.policy import AbstractPolicy

from .adversarial_policy import AdversarialPolicy


class JointPolicy(AdversarialPolicy):
    """Given a protagonist and an adversarial policy, combine to give a joint policy."""

    protagonist_policy: AbstractPolicy
    adversarial_policy: AbstractPolicy

    def __init__(
        self, protagonist_policy: AbstractPolicy, adversarial_policy: AbstractPolicy
    ) -> None:
        super().__init__(
            dim_state=protagonist_policy.dim_state,
            protagonist_dim_action=protagonist_policy.dim_action,
            adversarial_dim_action=adversarial_policy.dim_action,
            deterministic=protagonist_policy.deterministic,
            action_scale=torch.cat(
                (protagonist_policy.action_scale, adversarial_policy.action_scale)
            ),
        )
        self.protagonist_policy = protagonist_policy
        self.adversarial_policy = adversarial_policy

    def forward(self, state: State) -> TupleDistribution:
        """Forward compute the policy."""
        protagonist_mean, protagonist_chol = self.protagonist_policy(state)
        adversarial_mean, adversarial_chol = self.adversarial_policy(state)

        return self.stack_distributions(
            protagonist_mean, protagonist_chol, adversarial_mean, adversarial_chol
        )
