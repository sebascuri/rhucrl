"""Python Script Template."""
from importlib import import_module
from typing import Any, Optional, Type, TypeVar

from rllib.dataset.datatypes import Observation

from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.policy.split_policy import SplitPolicy

from .adversarial_agent import AdversarialAgent

T = TypeVar("T", bound="ZeroSumAgent")


class ZeroSumAgent(AdversarialAgent):
    """Zero-Sum Agent.

    Zero-Sum has two dependent agents.
    The protagonist receives (s, a, r, s') and the protagonist (s, a, -r, s').

    """

    policy: SplitPolicy

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        assert (
            self.protagonist_agent.policy.base_policy
            is self.antagonist_agent.policy.base_policy
        ), "Protagonist and Adversarial agent should share the base policy."
        self.policy = self.protagonist_agent.policy

    def observe(self, observation: Observation) -> None:
        """Send observations to both players.

        The protagonist receives (s, a, r, s') and the antagonist (s, a, -r, s').

        """
        super().observe(observation)
        protagonist_observation = Observation(*observation)
        antagonist_observation = Observation(*observation)
        antagonist_observation.reward = -observation.reward

        self.send_observations(protagonist_observation, antagonist_observation)

    @classmethod
    def default(
        cls: Type[T],
        environment: AdversarialEnv,
        protagonist_agent_name: str = "SAC",
        antagonist_agent_name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Get default Zero-Sum agent."""
        agent_ = getattr(import_module("rllib.agent"), f"{protagonist_agent_name}Agent")
        protagonist_agent = agent_.default(
            environment, comment="Protagonist", *args, **kwargs
        )

        if antagonist_agent_name is None:
            antagonist_agent_name = protagonist_agent_name
        agent_ = getattr(import_module("rllib.agent"), f"{antagonist_agent_name}Agent")
        antagonist_agent = agent_.default(
            environment, comment="Antagonist", *args, **kwargs
        )

        protagonist_policy = SplitPolicy(
            base_policy=protagonist_agent.policy,
            protagonist_dim_action=environment.protagonist_dim_action,
            antagonist_dim_action=environment.antagonist_dim_action,
            protagonist=True,
        )

        antagonist_policy = SplitPolicy(
            base_policy=protagonist_agent.policy,
            protagonist_dim_action=environment.protagonist_dim_action,
            antagonist_dim_action=environment.antagonist_dim_action,
            protagonist=False,
        )

        protagonist_agent.set_policy(protagonist_policy)
        antagonist_agent.set_policy(antagonist_policy)

        return super().default(
            environment,
            protagonist_agent=protagonist_agent,
            antagonist_agent=antagonist_agent,
            *args,
            **kwargs,
        )
