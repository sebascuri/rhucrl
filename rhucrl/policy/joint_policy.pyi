"""Python Script Template."""
from typing import Any, Tuple, Type, TypeVar

from rllib.dataset.datatypes import Action
from rllib.policy import AbstractPolicy

from rhucrl.environment.adversarial_environment import AdversarialEnv

from .adversarial_policy import AdversarialPolicy

T = TypeVar("T", bound="JointPolicy")

class JointPolicy(AdversarialPolicy):
    """Given a protagonist and an antagonist policy, combine to give a joint policy."""

    def __init__(
        self,
        dim_action: Tuple[int],
        action_scale: Action,
        protagonist_policy: AbstractPolicy,
        antagonist_policy: AbstractPolicy,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @classmethod
    def default(
        cls: Type[T],
        environment: AdversarialEnv,
        protagonist: bool = ...,
        antagonist: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
