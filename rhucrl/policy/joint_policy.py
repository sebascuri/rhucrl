"""Python Script Template."""

import torch
from rllib.policy import NNPolicy

from rhucrl.environment.utilities import (
    adversarial_to_antagonist_environment,
    adversarial_to_protagonist_environment,
)

from .adversarial_policy import AdversarialPolicy


class JointPolicy(AdversarialPolicy):
    """Given a protagonist and an antagonist policy, combine to give a joint policy."""

    def __init__(
        self,
        dim_action,
        action_scale,
        protagonist_policy,
        antagonist_policy,
        *args,
        **kwargs,
    ) -> None:
        assert protagonist_policy.deterministic == antagonist_policy.deterministic
        assert protagonist_policy.dim_state == antagonist_policy.dim_state
        super().__init__(
            *args,
            dim_state=protagonist_policy.dim_state,
            dim_action=dim_action,
            protagonist_dim_action=protagonist_policy.dim_action,
            antagonist_dim_action=antagonist_policy.dim_action,
            deterministic=protagonist_policy.deterministic,
            action_scale=action_scale,
            dist_params=protagonist_policy.dist_params,
            **kwargs,
        )  # type: ignore
        self.protagonist_policy = protagonist_policy
        self.antagonist_policy = antagonist_policy

    def forward(self, state):
        """Forward compute the policy."""
        p_dim = self.dim_action[0] - self.antagonist_dim_action[0]
        a_dim = self.dim_action[0] - self.protagonist_dim_action[0]

        p_mean, p_scale_tril = self.protagonist_policy(state)
        if self.only_protagonist:
            return p_mean, p_scale_tril

        a_mean, a_scale_tril = self.antagonist_policy(state)

        p_std = p_scale_tril.diagonal(dim1=-1, dim2=-2)
        a_std = a_scale_tril.diagonal(dim1=-1, dim2=-2)

        if self.protagonist or self.weak_antagonist:
            h_mean = p_mean[..., p_dim:]
            h_std = p_std[..., p_dim:]
        elif self.strong_antagonist:
            h_mean = a_mean[..., a_dim:]
            h_std = a_std[..., a_dim:]
        else:
            raise NotImplementedError

        pro_mean, pro_std = p_mean[..., :p_dim], p_std[..., :p_dim]
        ant_mean, ant_std = a_mean[..., :a_dim], a_std[..., :a_dim]

        mean = torch.cat((pro_mean, ant_mean, h_mean), dim=-1)
        std = torch.cat((pro_std, ant_std, h_std), dim=-1)
        return mean, std.diag_embed()

    @classmethod
    def default(
        cls,
        environment,
        protagonist=True,
        weak_antagonist=False,
        strong_antagonist=False,
        *args,
        **kwargs,
    ):
        """Get default policy."""
        protagonist_env = adversarial_to_protagonist_environment(environment)
        protagonist_policy = NNPolicy.default(protagonist_env, *args, **kwargs)

        antagonist_env = adversarial_to_antagonist_environment(environment)
        antagonist_policy = NNPolicy.default(antagonist_env, *args, **kwargs)

        return cls(
            dim_action=environment.dim_action,
            action_scale=environment.action_scale,
            protagonist_policy=protagonist_policy,
            antagonist_policy=antagonist_policy,
            protagonist=protagonist,
            weak_antagonist=weak_antagonist,
            strong_antagonist=strong_antagonist,
        )
