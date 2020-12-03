"""Action Robust policies.."""
from abc import ABCMeta

import torch
from rllib.policy.nn_policy import NNPolicy

from .split_policy import SplitPolicy


class ActionRobustPolicy(SplitPolicy, metaclass=ABCMeta):
    """Action Robust Abstract Policy class.

    Parameters
    ----------
    alpha: float
        Action robust parameter.

    References
    ----------
    Tessler, C., Efroni, Y., & Mannor, S. (2019).
    Action robust reinforcement learning and applications in continuous control. ICML.

    """

    def __init__(self, alpha, *args, **kwargs):
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def forward(self, state):
        """Forward compute the policy."""
        raise NotImplementedError

    @classmethod
    def default(cls, environment, hallucinate_protagonist=True, *args, **kwargs):
        """See `NNPolicy.default'."""
        protagonist_policy = NNPolicy(
            dim_state=environment.dim_state,
            dim_action=environment.protagonist_dim_action,
        )
        antagonist_policy = NNPolicy(
            dim_state=environment.dim_state,
            dim_action=environment.antagonist_dim_action,
        )
        assert environment.protagonist_dim_action == environment.antagonist_dim_action

        hallucination_policy = NNPolicy(
            dim_state=environment.dim_state, dim_action=environment.dim_state
        )
        return cls(
            alpha=environment.alpha,
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            protagonist_policy=protagonist_policy,
            antagonist_policy=antagonist_policy,
            hallucination_policy=hallucination_policy,
            hallucinate_protagonist=hallucinate_protagonist,
        )


class NoisyActionRobustPolicy(ActionRobustPolicy):
    """Noisy Action Robust Abstract Policy class.

    It averages both policies with weight 1-alpha/alpha.

    References
    ----------
    Tessler, C., Efroni, Y., & Mannor, S. (2019).
    Action robust reinforcement learning and applications in continuous control. ICML.
    """

    def forward(self, state):
        """Compute policy."""
        p_mean, p_scale_tril = self.protagonist_policy(state)
        a_mean, a_scale_tril = self.antagonist_policy(state)
        h_mean, h_scale_tril = self.hallucination_policy(state)

        p_std = p_scale_tril.diagonal(dim1=-1, dim2=-2)
        a_std = a_scale_tril.diagonal(dim1=-1, dim2=-2)
        h_std = h_scale_tril.diagonal(dim1=-1, dim2=-2)

        mean = (1 - self.alpha) * p_mean + self.alpha * a_mean
        std = (1 - self.alpha) * p_std + self.alpha * a_std

        return self.stack_policies((mean, h_mean), (std, h_std))


class ProbabilisticActionRobustPolicy(ActionRobustPolicy):
    """Noisy Action Robust Abstract Policy class.

    It samples the protagonist with probability 1-alpha and the antagonist with alpha.

    References
    ----------
    Tessler, C., Efroni, Y., & Mannor, S. (2019).
    Action robust reinforcement learning and applications in continuous control. ICML.
    """

    def forward(self, state):
        """Compute policy."""
        h_mean, h_scale_tril = self.hallucination_policy(state)
        if torch.rand() < self.alpha:
            mean, scale_tril = self.antagonist_policy(state)
        else:
            mean, scale_tril = self.protagonist_policy(state)
        std = scale_tril.diagonal(dim1=-1, dim2=-2)
        h_std = h_scale_tril.diagonal(dim1=-1, dim2=-2)

        return self.stack_policies((mean, h_mean), (std, h_std))
