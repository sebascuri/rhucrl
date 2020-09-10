"""Python Script Template."""
from typing import Any, Tuple

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Action

from rhucrl.environment.adversarial_environment import AdversarialEnv

from rhucrl.policy.joint_policy import JointPolicy

from .adversarial_agent import AdversarialAgent

class RARLAgent(AdversarialAgent):
    policy: JointPolicy
    def __init__(
        self, dim_action: Tuple[int], action_scale: Action, *args: Any, **kwargs: Any
    ) -> None: ...
    @staticmethod
    def get_default_protagonist(
        environment: AdversarialEnv,
        protagonist_agent_name: str = ...,
        *args: Any,
        **kwargs: Any,
    ) -> AbstractAgent: ...
    @staticmethod
    def get_default_antagonist(
        environment: AdversarialEnv,
        antagonist_agent_name: str = ...,
        *args: Any,
        **kwargs: Any,
    ) -> AbstractAgent: ...
    @staticmethod
    def get_default_agents(
        environment: AdversarialEnv, *args: Any, **kwargs: Any
    ) -> Tuple[AbstractAgent, AbstractAgent]: ...
