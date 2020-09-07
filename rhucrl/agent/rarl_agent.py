"""Python Script Template."""
from importlib import import_module

from rllib.dataset.datatypes import Observation

from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.environment.utilities import (
    adversarial_to_adversary_environment,
    adversarial_to_protagonist_environment,
)

from .adversarial_agent import AdversarialAgent


class RARLAgent(AdversarialAgent):
    """RARL Agent.

    RARL has two independent agents.
    The protagonist receives (s, a_pro, r, s') and the adversary (s, a_adv, -r, s').

    """

    def observe(self, observation: Observation) -> None:
        """Send observations to both players.

        This is the crucial method as it needs to separate the actions.
        """
        super().observe(observation)
        protagonist_dim_action = self.policy.protagonist_dim_action[0]
        protagonist_observation = Observation(*observation)
        protagonist_observation.action = observation.action[:protagonist_dim_action]

        adversarial_observation = Observation(*observation)
        adversarial_observation.action = observation.action[protagonist_dim_action:]
        adversarial_observation.reward = -observation.reward

        self.send_observations(protagonist_observation, adversarial_observation)

    @classmethod
    def default(
        cls,
        environment: AdversarialEnv,
        protagonist_agent_name: str = "SAC",
        adversarial_agent_name: str = "SAC",
        *args,
        **kwargs,
    ):
        """Get default RARL agent."""
        protagonist_environment = adversarial_to_protagonist_environment(environment)
        adversary_environment = adversarial_to_adversary_environment(environment)

        protagonist_agent = getattr(
            import_module("rllib.agent"), f"{protagonist_agent_name}Agent"
        ).default(protagonist_environment, *args, **kwargs)

        adversary_agent = getattr(
            import_module("rllib.agent"), f"{adversarial_agent_name}Agent"
        ).default(adversary_environment, *args, **kwargs)

        return cls(
            protagonist_agent,
            adversary_agent,
            train_frequency=protagonist_agent.train_frequency,
            num_iter=protagonist_agent.num_iter,
            num_rollouts=protagonist_agent.num_rollouts,
            *args,
            **kwargs,
        )
