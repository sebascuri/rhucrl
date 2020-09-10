"""Python Script Template."""

import torch
from rllib.policy import NNPolicy
from rllib.util.neural_networks.utilities import deep_copy_module

from rhucrl.environment.utilities import (
    adversarial_to_antagonist_environment,
    adversarial_to_protagonist_environment,
)

from .adversarial_policy import AdversarialPolicy


class SplitPolicy(AdversarialPolicy):
    """Split a policy into protagonist and antagonist policies."""

    def __init__(
        self,
        base_policy,
        protagonist_dim_action,
        antagonist_dim_action,
        *args,
        **kwargs,
    ):
        super().__init__(
            protagonist_dim_action=protagonist_dim_action,
            antagonist_dim_action=antagonist_dim_action,
            dim_state=base_policy.dim_state,
            dim_action=base_policy.dim_action,
            deterministic=base_policy.deterministic,
            action_scale=base_policy.action_scale,
            dist_params=base_policy.dist_params,
            *args,
            **kwargs,
        )
        self.base_policy = base_policy
        self.strong_antagonist_policy = deep_copy_module(self.base_policy)

    def forward(self, state):
        """Forward compute the policy."""
        p_dim = self.protagonist_dim_action[0]
        a_dim = self.antagonist_dim_action[0]
        r_dim = p_dim + a_dim
        pwa_mean, pwa_scale_tril = self.base_policy(state)
        if self.only_protagonist:
            return pwa_mean, pwa_scale_tril

        pwa_std = pwa_scale_tril.diagonal(dim1=-1, dim2=-2)
        p_mean, p_std = pwa_mean[..., :p_dim], pwa_std[..., :p_dim]

        if self.protagonist or self.weak_antagonist:
            a_mean, a_std = pwa_mean[..., p_dim:r_dim], pwa_std[..., p_dim:r_dim]
            h_mean, h_std = pwa_mean[..., r_dim:], pwa_std[..., r_dim:]
        else:
            sa_mean, sa_scale_tril = self.strong_antagonist_policy(state)
            sa_std = sa_scale_tril.diagonal(dim1=-1, dim2=-2)
            a_mean, a_std = sa_mean[..., p_dim:r_dim], sa_std[..., p_dim:r_dim]
            h_mean, h_std = sa_mean[..., r_dim:], sa_std[..., r_dim:]

        if self.protagonist:
            a_mean, a_std = a_mean.detach(), a_std.detach()
        elif self.weak_antagonist:
            p_mean, p_std = p_mean.detach(), p_std.detach()
            h_mean, h_std = h_mean.detach(), h_std.detach()
        elif self.strong_antagonist:
            p_mean, p_std = p_mean.detach(), p_std.detach()
        else:
            raise NotImplementedError

        mean = torch.cat((p_mean, a_mean, h_mean), dim=-1)
        std = torch.cat((p_std, a_std, h_std), dim=-1)
        return mean, std.diag_embed()

    @classmethod
    def default(
        cls,
        environment,
        base_policy=None,
        protagonist=True,
        weak_antagonist=False,
        strong_antagonist=False,
        *args,
        **kwargs,
    ):
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
            protagonist=protagonist,
            weak_antagonist=weak_antagonist,
            strong_antagonist=strong_antagonist,
        )
