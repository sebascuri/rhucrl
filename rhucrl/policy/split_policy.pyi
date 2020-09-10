"""Python Script Template."""
from typing import Any, Optional, Type, TypeVar

from rllib.policy import AbstractPolicy

from rhucrl.environment.adversarial_environment import AdversarialEnv

from .adversarial_policy import AdversarialPolicy

T = TypeVar("T", bound="SplitPolicy")

class SplitPolicy(AdversarialPolicy):
    """Split a policy into protagonist and antagonist policies."""

    base_policy: AbstractPolicy
    def __init__(
        self, base_policy: AbstractPolicy, *args: Any, **kwargs: Any,
    ) -> None: ...
    @classmethod
    def default(
        cls: Type[T],
        environment: AdversarialEnv,
        base_policy: Optional[AbstractPolicy] = None,
        protagonist: bool = True,
        weak_antagonist: bool = False,
        strong_antagonist: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
