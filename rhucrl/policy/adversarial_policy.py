"""Python Script Template."""
from abc import ABCMeta

import torch
from rllib.policy import AbstractPolicy


class AdversarialPolicy(AbstractPolicy, metaclass=ABCMeta):
    """Given a protagonist and an antagonist policy, combine to give a joint policy."""

    def __init__(
        self,
        protagonist_dim_action,
        antagonist_dim_action,
        protagonist=True,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.protagonist_dim_action = protagonist_dim_action
        self.antagonist_dim_action = antagonist_dim_action

        self._protagonist = protagonist
        self._only_protagonist = False

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
    def only_protagonist(self):
        """Return flag that indicates that only protagonist is used."""
        return self._only_protagonist

    @only_protagonist.setter
    def only_protagonist(self, only_protagonist):
        """Set if only the protagonist is used to compute the policy."""
        self._only_protagonist = only_protagonist

    def stack_policies(self, means, stds):
        """Stack a set of policies."""
        mean = torch.cat(means, dim=-1)[..., : self.dim_action[0]]
        std = torch.cat(stds, dim=-1)[..., : self.dim_action[0]]
        return mean, std.diag_embed()

    def forward(self, state):
        """Forward compute the policy."""
        raise NotImplementedError


class ProtagonistPolicy(object):
    """Context Manager to set the policy to be protagonist."""

    def __init__(self, policy, protagonist=True):
        self.policy = policy
        self.old = self.policy.protagonist
        self.protagonist = protagonist

    def __enter__(self):
        """Enter into a Hallucination Context."""
        self.policy.protagonist = self.protagonist

    def __exit__(self, *args):
        """Exit the Hallucination Context."""
        self.policy.protagonist = self.old


class AntagonistPolicy(ProtagonistPolicy):
    """Context Manager to set the policy to be protagonist."""

    def __init__(self, policy, protagonist=False):
        super().__init__(policy, protagonist=protagonist)
