"""Python Script Template."""
from typing import Any, Optional, Type, TypeVar

from rllib.policy import AbstractPolicy

from rhucrl.environment.adversarial_environment import AdversarialEnv

from .adversarial_policy import AdversarialPolicy

class SplitPolicy(AdversarialPolicy):
    """Split a policy into protagonist and antagonist policies."""

    _hallucinate_protagonist: bool
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def hallucinate_protagonist(self) -> bool: ...
    @property
    def hallucinate_antagonist(self) -> bool: ...
