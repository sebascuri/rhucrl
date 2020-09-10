"""Python Script Template."""
from typing import Any, Tuple

from rllib.dataset.datatypes import TupleDistribution
from rllib.policy import AbstractPolicy
from torch import Tensor

class AdversarialPolicy(AbstractPolicy):
    """Given a protagonist and an antagonist policy, combine to give a joint policy."""

    protagonist_dim_action: Tuple[int]
    antagonist_dim_action: Tuple[int]

    _only_protagonist: bool
    def __init__(
        self,
        protagonist_dim_action: Tuple[int] = ...,
        antagonist_dim_action: Tuple[int] = ...,
        protagonist: bool = ...,
        weak_antagonist: bool = ...,
        strong_antagonist: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def only_protagonist(self) -> bool: ...
    @only_protagonist.setter
    def only_protagonist(self, only_protagonist: bool) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
