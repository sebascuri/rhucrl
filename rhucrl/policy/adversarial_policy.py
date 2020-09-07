"""Python Script Template."""
from typing import Tuple

import torch
from rllib.dataset.datatypes import State, TupleDistribution
from rllib.policy import AbstractPolicy
from rllib.util.neural_networks.utilities import get_batch_size


class AdversarialPolicy(AbstractPolicy):
    """Given a protagonist and an antagonist policy, combine to give a joint policy."""

    protagonist_dim_action: Tuple[int]
    antagonist_dim_action: Tuple[int]

    _only_protagonist: bool

    def __init__(
        self,
        dim_state: Tuple[int],
        protagonist_dim_action: Tuple[int],
        antagonist_dim_action: Tuple[int],
        *args,
        **kwargs
    ) -> None:
        dim_action = protagonist_dim_action[0] + antagonist_dim_action[0]

        super().__init__(dim_state=dim_state, dim_action=(dim_action,), *args, **kwargs)
        self.protagonist_dim_action = protagonist_dim_action
        self.antagonist_dim_action = antagonist_dim_action

        self._only_protagonist = False

    @property
    def only_protagonist(self) -> bool:
        """Return flag that indicates that only protagonist is used."""
        return self._only_protagonist

    @only_protagonist.setter
    def only_protagonist(self, only_protagonist: bool) -> None:
        """Set if only the protagonist is used to compute the policy."""
        self._only_protagonist = only_protagonist

    def stack_distributions(
        self,
        protagonist_mean: torch.Tensor,
        protagonist_chol: torch.Tensor,
        antagonist_mean: torch.Tensor,
        antagonist_chol: torch.Tensor,
    ) -> TupleDistribution:
        """Stack Protagonist and Adversarial distributions."""
        if self.only_protagonist:
            return protagonist_mean, protagonist_chol

        bs = get_batch_size(protagonist_mean, self.protagonist_dim_action)
        mean = torch.cat((protagonist_mean, antagonist_mean), dim=-1)
        if self.deterministic:
            chol = torch.zeros(bs + self.dim_action)
        else:
            chol = torch.zeros(bs + self.dim_action + self.dim_action)
            chol[
                ..., : self.protagonist_dim_action[0], : self.protagonist_dim_action[0]
            ] = protagonist_chol
            chol[
                ..., self.protagonist_dim_action[0] :, self.protagonist_dim_action[0] :
            ] = antagonist_chol

        return mean, chol

    def forward(self, state: State) -> TupleDistribution:
        """Forward compute the policy."""
        raise NotImplementedError
