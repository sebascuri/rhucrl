"""Python Script Template."""
from abc import ABCMeta

from rllib.policy import AbstractPolicy


class AdversarialPolicy(AbstractPolicy, metaclass=ABCMeta):
    """Given a protagonist and an antagonist policy, combine to give a joint policy."""

    def __init__(
        self,
        protagonist_dim_action=(),
        antagonist_dim_action=(),
        protagonist=True,
        weak_antagonist=False,
        strong_antagonist=False,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.protagonist_dim_action = protagonist_dim_action
        self.antagonist_dim_action = antagonist_dim_action

        self.protagonist = protagonist
        self.weak_antagonist = weak_antagonist
        self.strong_antagonist = strong_antagonist
        assert protagonist ^ weak_antagonist ^ strong_antagonist

        self._only_protagonist = False

    @property
    def only_protagonist(self):
        """Return flag that indicates that only protagonist is used."""
        return self._only_protagonist

    @only_protagonist.setter
    def only_protagonist(self, only_protagonist):
        """Set if only the protagonist is used to compute the policy."""
        self._only_protagonist = only_protagonist

    def forward(self, state):
        """Forward compute the policy."""
        raise NotImplementedError
