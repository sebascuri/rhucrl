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
        antagonist: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def protagonist(self) -> bool: ...
    @protagonist.setter
    def protagonist(self, new_value: bool) -> None: ...
    @property
    def antagonist(self) -> bool: ...
    @antagonist.setter
    def antagonist(self, new_value: bool) -> None: ...
    @property
    def only_protagonist(self) -> bool: ...
    @only_protagonist.setter
    def only_protagonist(self, only_protagonist: bool) -> None: ...
    def stack_policies(
        self, means: Tuple[Tensor], stds: Tuple[Tensor]
    ) -> TupleDistribution: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...

class ProtagonistPolicy(object):
    policy: AbstractPolicy
    old: bool
    protagonist: bool
    def __init__(self, policy: AbstractPolicy, protagonist: bool = ...) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...

class AntagonistPolicy(ProtagonistPolicy): ...
