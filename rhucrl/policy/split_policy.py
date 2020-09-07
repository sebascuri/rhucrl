"""Python Script Template."""
from typing import Any, Optional, Tuple, Type, TypeVar

from rllib.dataset.datatypes import State, TupleDistribution
from rllib.policy import AbstractPolicy, NNPolicy

from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.environment.utilities import (
    adversarial_to_antagonist_environment,
    adversarial_to_protagonist_environment,
)

from .adversarial_policy import AdversarialPolicy

T = TypeVar("T", bound="SplitPolicy")


class SplitPolicy(AdversarialPolicy):
    """Split a policy into protagonist and antagonist policies."""

    base_policy: AbstractPolicy

    def __init__(
        self,
        base_policy: AbstractPolicy,
        protagonist_dim_action: Tuple[int],
        antagonist_dim_action: Tuple[int],
        protagonist: bool = True,
    ) -> None:
        super().__init__(
            dim_state=base_policy.dim_state,
            protagonist_dim_action=protagonist_dim_action,
            antagonist_dim_action=antagonist_dim_action,
            deterministic=base_policy.deterministic,
            action_scale=base_policy.action_scale,
            dist_params=base_policy.dist_params,
        )
        self.base_policy = base_policy
        self._protagonist = protagonist

    def forward(self, state: State) -> TupleDistribution:
        """Forward compute the policy."""
        joint_mean, joint_chol = self.base_policy(state)

        protagonist_mean = joint_mean[..., : self.protagonist_dim_action[0]]
        antagonist_mean = joint_mean[..., self.protagonist_dim_action[0] :]

        protagonist_chol = joint_chol[
            ..., : self.protagonist_dim_action[0], : self.protagonist_dim_action[0]
        ]

        antagonist_chol = joint_chol[
            ..., self.protagonist_dim_action[0] :, self.protagonist_dim_action[0] :
        ]

        if self.only_protagonist:
            return protagonist_mean, protagonist_chol

        if self._protagonist:
            antagonist_mean = antagonist_mean.detach()
            antagonist_chol = antagonist_chol.detach()
        else:
            protagonist_mean = protagonist_mean.detach()
            protagonist_chol = protagonist_chol.detach()

        return self.stack_distributions(
            protagonist_mean, protagonist_chol, antagonist_mean, antagonist_chol
        )

    @classmethod
    def default(
        cls: Type[T],
        environment: AdversarialEnv,
        base_policy: Optional[AbstractPolicy] = None,
        protagonist: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Get default policy."""
        if protagonist:
            derived_env = adversarial_to_protagonist_environment(environment)
        else:
            derived_env = adversarial_to_antagonist_environment(environment)

        if base_policy is None:
            base_policy = NNPolicy.default(derived_env, *args, **kwargs)

        return cls(
            base_policy,
            environment.protagonist_dim_action,
            environment.antagonist_dim_action,
            protagonist,
        )
