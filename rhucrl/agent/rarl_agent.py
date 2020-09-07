"""Python Script Template."""
from importlib import import_module
from typing import Any, Optional

from rllib.dataset.datatypes import Observation

from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.environment.utilities import (
    adversarial_to_antagonist_environment,
    adversarial_to_protagonist_environment,
)
from rhucrl.policy.joint_policy import JointPolicy

from .adversarial_agent import AdversarialAgent


class RARLAgent(AdversarialAgent):
    """RARL Agent.

    RARL has two independent agents.
    The protagonist receives (s, a_pro, r, s') and the antagonist (s, a_ant, -r, s').

    """

    policy: JointPolicy

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.policy = JointPolicy(
            self.protagonist_agent.policy, self.antagonist_agent.policy
        )

    def observe(self, observation: Observation) -> None:
        """Send observations to both players.

        This is the crucial method as it needs to separate the actions.
        """
        super().observe(observation)
        protagonist_dim_action = self.policy.protagonist_dim_action[0]
        protagonist_observation = Observation(*observation)
        protagonist_observation.action = observation.action[:protagonist_dim_action]

        antagonist_observation = Observation(*observation)
        antagonist_observation.action = observation.action[protagonist_dim_action:]
        antagonist_observation.reward = -observation.reward

        self.send_observations(protagonist_observation, antagonist_observation)

    @classmethod
    def default(
        cls,
        environment: AdversarialEnv,
        protagonist_agent_name: str = "SAC",
        antagonist_agent_name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Get default RARL agent."""
        protagonist_environment = adversarial_to_protagonist_environment(environment)
        antagonist_environment = adversarial_to_antagonist_environment(environment)

        protagonist_agent = getattr(
            import_module("rllib.agent"), f"{protagonist_agent_name}Agent"
        ).default(protagonist_environment, comment="Protagonist", *args, **kwargs)

        if antagonist_agent_name is None:
            antagonist_agent_name = protagonist_agent_name
        antagonist_agent = getattr(
            import_module("rllib.agent"), f"{antagonist_agent_name}Agent"
        ).default(antagonist_environment, comment="Antagonist", *args, **kwargs)

        return super().default(
            environment,
            protagonist_agent=protagonist_agent,
            antagonist_agent=antagonist_agent,
            *args,
            **kwargs,
        )
