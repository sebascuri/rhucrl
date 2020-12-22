"""Python Script Template."""
from abc import ABCMeta

import torch
from rllib.algorithms.pathwise_loss import PathwiseLoss
from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function.nn_ensemble_value_function import NNEnsembleQFunction

from rhucrl.policy.action_robust_policy import (
    ActionRobustPolicy,
    NoisyActionRobustPolicy,
    ProbabilisticActionRobustPolicy,
)


class ActionRobustPathwiseLoss(PathwiseLoss, metaclass=ABCMeta):
    """Pathwise Loss for Action Robust RL."""

    policy: ActionRobustPolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = self.policy.alpha

    @staticmethod
    def _get_action(policy, state):
        pi = tensor_to_distribution(policy(state), **policy.dist_params)
        return policy.action_scale * pi.rsample().clamp(-1, 1)

    def _add_hallucination_action(self, state, protagonist_action, antagonist_action):
        hallucination_action = self._get_action(self.policy.hallucination_policy, state)

        if self.policy.hallucinate_protagonist:
            protagonist_action = torch.cat(
                (protagonist_action, hallucination_action), dim=-1
            )
            antagonist_action = torch.cat(
                (antagonist_action, hallucination_action.detach()), dim=-1
            )
        else:
            protagonist_action = torch.cat(
                (protagonist_action, hallucination_action.detach()), dim=-1
            )
            antagonist_action = torch.cat(
                (antagonist_action, hallucination_action), dim=-1
            )
        return (
            protagonist_action[..., : self.critic.dim_action[0]],
            antagonist_action[..., : self.critic.dim_action[0]],
        )

    def _loss(self, state, action):
        with DisableGradient(self.critic):
            q = self.critic(state, action)
            if isinstance(self.critic, NNEnsembleQFunction):
                q = q[..., 0]
        return -q

    def forward(self, observation):
        """Compute path-wise loss."""
        raise NotImplementedError


class NoisyActionRobustPathwiseLoss(ActionRobustPathwiseLoss):
    """Get Noisy robust Patwise Loss."""

    policy: NoisyActionRobustPolicy

    def forward(self, observation):
        """Compute path-wise loss."""
        if self.policy is None or self.critic is None:
            return Loss()
        state = observation.state

        protagonist_action_ = self._get_action(self.policy.protagonist_policy, state)
        antagonist_action_ = self._get_action(self.policy.antagonist_policy, state)
        protagonist_action = (
            1 - self.alpha
        ) * protagonist_action_ + self.alpha * antagonist_action_.detach()
        antagonist_action = (
            1 - self.alpha
        ) * protagonist_action_.detach() + self.alpha * antagonist_action_
        protagonist_action, antagonist_action = self._add_hallucination_action(
            state=state,
            protagonist_action=protagonist_action,
            antagonist_action=antagonist_action,
        )

        protagonist_loss = self._loss(state, protagonist_action)
        antagonist_loss = self._loss(state, antagonist_action)

        return Loss(policy_loss=protagonist_loss - antagonist_loss)


class ProbabilisticActionRobustPathwiseLoss(ActionRobustPathwiseLoss):
    """Get Probabilistic Robust Patwise Loss."""

    policy: ProbabilisticActionRobustPolicy

    def forward(self, observation):
        """Compute path-wise loss."""
        if self.policy is None or self.critic is None:
            return Loss()
        state = observation.state

        protagonist_action = self._get_action(self.policy.protagonist_policy, state)
        antagonist_action = self._get_action(self.policy.antagonist_policy, state)

        protagonist_action, antagonist_action = self._add_hallucination_action(
            state=state,
            protagonist_action=protagonist_action,
            antagonist_action=antagonist_action,
        )

        protagonist_loss = (1 - self.alpha) * self._loss(state, protagonist_action)
        antagonist_loss = self.alpha * self._loss(state, antagonist_action)
        return Loss(policy_loss=protagonist_loss - antagonist_loss)
