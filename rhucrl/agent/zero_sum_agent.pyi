from typing import Any, Tuple

from rllib.agent import AbstractAgent
from rllib.policy import AbstractPolicy

from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.policy.split_policy import SplitPolicy

from .adversarial_agent import AdversarialAgent

class ZeroSumAgent(AdversarialAgent):
    """Zero-Sum Agent.

    Zero-Sum has two dependent agents.
    The protagonist receives (s, a, r, s') and the protagonist (s, a, -r, s').

    """

    policy: SplitPolicy
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @staticmethod
    def get_default_protagonist(
        environment: AdversarialEnv,
        protagonist_name: str = ...,
        *args: Any,
        **kwargs: Any,
    ) -> AbstractAgent: ...
    @staticmethod
    def get_default_antagonist(
        environment: AdversarialEnv,
        base_policy: AbstractPolicy,
        antagonist_name: str = ...,
        *args: Any,
        **kwargs: Any,
    ) -> AbstractAgent: ...
