"""Python Script Template."""
from abc import ABCMeta

import torch
from hucrl.policy.augmented_policy import AugmentedPolicy
from rllib.policy import AbstractPolicy, NNPolicy


class AdversarialPolicy(AbstractPolicy, metaclass=ABCMeta):
    """Given a protagonist and an antagonist policy, combine to give a joint policy."""

    def __init__(
        self,
        protagonist_policy,
        antagonist_policy,
        hallucination_policy=None,
        protagonist=True,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(
            deterministic=protagonist_policy.deterministic,
            dist_params=protagonist_policy.dist_params,
            *args,
            **kwargs,
        )
        self._protagonist_policy = protagonist_policy
        self._antagonist_policy = antagonist_policy
        assert protagonist_policy.dim_state == antagonist_policy.dim_state
        assert protagonist_policy.dim_state == self.dim_state

        self._protagonist = protagonist
        if hallucination_policy is None:
            hallucination_policy = NNPolicy(
                dim_state=self.dim_state, dim_action=self.dim_action
            )
        self._hallucination_policy = hallucination_policy

    @property
    def protagonist_policy(self):
        """Return protagonist policy."""
        if isinstance(self._protagonist_policy, AugmentedPolicy):
            return self._protagonist_policy.true_policy
        else:
            return self._protagonist_policy

    def set_protagonist_policy(self, new_policy):
        """Set protagonist policy."""
        if isinstance(self._protagonist_policy, AugmentedPolicy):
            self._protagonist_policy.true_policy = new_policy
        else:
            self._protagonist_policy = new_policy

    @property
    def antagonist_policy(self):
        """Return antagonist policy."""
        if isinstance(self._antagonist_policy, AugmentedPolicy):
            return self._antagonist_policy.true_policy
        else:
            return self._antagonist_policy

    def set_antagonist_policy(self, new_policy):
        """Set antagonist policy."""
        if isinstance(self._antagonist_policy, AugmentedPolicy):
            self._antagonist_policy.true_policy = new_policy
        else:
            self._antagonist_policy = new_policy

    @property
    def hallucination_policy(self):
        """Return hallucination policy."""
        if isinstance(self._protagonist_policy, AugmentedPolicy) and self.protagonist:
            return self._protagonist_policy.hallucination_policy
        elif isinstance(self._antagonist_policy, AugmentedPolicy) and self.antagonist:
            return self._antagonist_policy.hallucination_policy
        else:
            return self._hallucination_policy

    def set_hallucination_policy(self, new_policy):
        """Set hallucination policy."""
        if isinstance(self._protagonist_policy, AugmentedPolicy):
            if self.protagonist:
                self._protagonist_policy.hallucination_policy = new_policy
            else:
                self._antagonist_policy.true_policy = new_policy
        else:
            self._hallucination_policy = new_policy

    @property
    def protagonist(self):
        """Return true if it is in protagonist mode."""
        return self._protagonist

    @protagonist.setter
    def protagonist(self, new_value):
        """Set protagonist value."""
        self._protagonist = new_value

    @property
    def antagonist(self):
        """Return true if it is in antagonist mode."""
        return not self._protagonist

    @antagonist.setter
    def antagonist(self, new_value):
        """Set protagonist value."""
        self._protagonist = not new_value

    @property
    def deterministic(self):
        """Get flag if the policy is deterministic or not."""
        return self._deterministic

    @deterministic.setter
    def deterministic(self, value):
        """Set flag if the policy is deterministic or not."""
        self._deterministic = value
        self.protagonist_policy.deterministic = value
        self.antagonist_policy.deterministic = value
        self.hallucination_policy.deterministic = value

    @torch.jit.export
    def reset(self):
        """Reset policy parameters (for example internal states)."""
        super().reset()
        self.protagonist_policy.reset()
        self.antagonist_policy.reset()
        self.hallucination_policy.reset()

    @torch.jit.export
    def update(self):
        """Update policy parameters."""
        super().update()
        self.protagonist_policy.update()
        self.antagonist_policy.update()
        self.hallucination_policy.update()

    @torch.jit.export
    def set_goal(self, goal=None):
        """Set policy goal."""
        super().set_goal(goal)
        self.protagonist_policy.set_goal(goal)
        self.antagonist_policy.set_goal(goal)
        self.hallucination_policy.set_goal(goal)

    def stack_policies(self, means, stds):
        """Stack a set of policies."""
        mean = torch.cat(means, dim=-1)[..., : self.dim_action[0]]
        std = torch.cat(stds, dim=-1)[..., : self.dim_action[0]]
        return mean, std.diag_embed()

    def forward(self, state):
        """Forward compute the policy."""
        raise NotImplementedError
