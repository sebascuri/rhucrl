"""Python Script Template."""
from typing import Any, Type, TypeVar

from rllib.policy import AbstractPolicy

from rhucrl.environment import AdversarialEnv

from .adversarial_policy import AdversarialPolicy

T = TypeVar("T", bound="SplitPolicy")

class SplitPolicy(AdversarialPolicy):
    """Split a policy into protagonist and antagonist policies."""

    hallucination_policy: AbstractPolicy
    _hallucinate_protagonist: bool
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def hallucinate_protagonist(self) -> bool: ...
    @property
    def hallucinate_antagonist(self) -> bool: ...
    @classmethod
    def default(
        cls: Type[T],
        environment: AdversarialEnv,
        protagonist: bool = ...,
        antagonist: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
    @classmethod
    def from_adversarial_policy(
        cls: Type[T], adversarial_policy: AdversarialPolicy, *args: Any, **kwargs: Any
    ) -> T: ...
