"""Python Script Template."""
from typing import Tuple

from rllib.dataset.datatypes import State, TupleDistribution
from rllib.policy import AbstractPolicy

from .adversarial_policy import AdversarialPolicy


class SplitPolicy(AdversarialPolicy):
    """Split a policy into protagonist and adversarial policies."""

    base_policy: AbstractPolicy
    adversarial_policy: AbstractPolicy

    def __init__(
        self,
        base_policy: AbstractPolicy,
        protagonist_dim_action: Tuple[int],
        adversarial_dim_action: Tuple[int],
        protagonist: bool = True,
    ) -> None:
        super().__init__(
            dim_state=base_policy.dim_state,
            protagonist_dim_action=protagonist_dim_action,
            adversarial_dim_action=adversarial_dim_action,
            deterministic=base_policy.deterministic,
            action_scale=base_policy.action_scale,
        )
        self.base_policy = base_policy
        self._protagonist = protagonist

    def forward(self, state: State) -> TupleDistribution:
        """Forward compute the policy."""
        joint_mean, joint_chol = self.base_policy(state)

        protagonist_mean = joint_mean[..., : self.protagonist_dim_action[0]]
        adversarial_mean = joint_mean[..., self.protagonist_dim_action[0] :]

        protagonist_chol = joint_chol[
            ..., : self.protagonist_dim_action[0], : self.protagonist_dim_action[0]
        ]

        adversarial_chol = joint_chol[
            ..., self.protagonist_dim_action[0] :, self.protagonist_dim_action[0] :
        ]

        if self.only_protagonist:
            return protagonist_mean, protagonist_chol

        if self._protagonist:
            adversarial_mean = adversarial_mean.detach()
            adversarial_chol = adversarial_chol.detach()
        else:
            protagonist_mean = protagonist_mean.detach()
            protagonist_chol = protagonist_chol.detach()

        return self.stack_distributions(
            protagonist_mean, protagonist_chol, adversarial_mean, adversarial_chol
        )
