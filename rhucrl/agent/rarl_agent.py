"""Python Script Template."""
from importlib import import_module

from rllib.dataset.datatypes import Observation

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

    def __init__(self, dim_action, action_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = JointPolicy(
            dim_action,
            action_scale,
            protagonist_policy=self.protagonist_agent.policy,
            antagonist_policy=self.antagonist_agent.policy,
        )

    def observe(self, observation) -> None:
        """Send observations to both players.

        This is the crucial method as it needs to separate the actions.
        """
        super().observe(observation)
        protagonist_dim_action = self.policy.protagonist_dim_action[0]
        protagonist_observation = Observation(*tuple(o.clone() for o in observation))
        protagonist_observation.action = observation.action[:protagonist_dim_action]

        antagonist_observation = Observation(*observation)
        antagonist_observation.action = observation.action[protagonist_dim_action:]
        antagonist_observation.reward = -observation.reward

        self.send_observations(protagonist_observation, antagonist_observation)

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """Get default RARL agent."""
        p_agent, a_agent = RARLAgent.get_default_agents(environment, *args, **kwargs)

        return super().default(
            environment,
            dim_action=environment.dim_action,
            action_scale=environment.action_scale,
            protagonist_agent=p_agent,
            antagonist_agent=a_agent,
            *args,
            **kwargs,
        )

    @staticmethod
    def get_default_protagonist(environment, protagonist_name="SAC", *args, **kwargs):
        """Get protagonist using RARL."""
        protagonist_environment = adversarial_to_protagonist_environment(environment)
        protagonist_agent = getattr(
            import_module("rllib.agent"), f"{protagonist_name}Agent"
        ).default(protagonist_environment, comment="Protagonist", *args, **kwargs)
        return protagonist_agent

    @staticmethod
    def get_default_antagonist(environment, antagonist_name="SAC", *args, **kwargs):
        """Get protagonist using RARL."""
        protagonist_environment = adversarial_to_antagonist_environment(environment)
        antagonist_agent = getattr(
            import_module("rllib.agent"), f"{antagonist_name}Agent"
        ).default(protagonist_environment, comment="Antagonist", *args, **kwargs)
        return antagonist_agent

    @staticmethod
    def get_default_agents(environment, *args, **kwargs):
        """Get default RARL agent."""
        p_agent = RARLAgent.get_default_protagonist(environment, *args, **kwargs)
        a_agent = RARLAgent.get_default_antagonist(environment, *args, **kwargs)
        return p_agent, a_agent
