"""Python Script Template."""
from importlib import import_module

from rllib.dataset.datatypes import Observation
from rllib.util.neural_networks.utilities import deep_copy_module

from rhucrl.environment.utilities import (
    adversarial_to_antagonist_environment,
    adversarial_to_protagonist_environment,
)
from rhucrl.model import HallucinatedModel
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
            dim_action=dim_action,
            action_scale=action_scale,
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
    def default(
        cls,
        environment,
        protagonist_dynamical_model=None,
        antagonist_dynamical_model=None,
        *args,
        **kwargs,
    ):
        """Get default RARL agent."""
        p_agent = RARLAgent.get_default_protagonist(
            environment,
            dynamical_model=deep_copy_module(protagonist_dynamical_model),
            *args,
            **kwargs,
        )
        a_agent = RARLAgent.get_default_antagonist(
            environment,
            dynamical_model=deep_copy_module(antagonist_dynamical_model),
            *args,
            **kwargs,
        )

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
        p_env = adversarial_to_protagonist_environment(environment)
        p_a = getattr(import_module("rllib.agent"), f"{protagonist_name}Agent").default(
            p_env, comment="Protagonist", *args, **kwargs
        )
        if hasattr(p_a, "dynamical_model"):
            p_a.dynamical_model.dim_action = p_env.protagonist_dim_action

        return p_a

    @staticmethod
    def get_default_antagonist(environment, antagonist_name="SAC", *args, **kwargs):
        """Get protagonist using RARL."""
        strong_antagonist = isinstance(
            kwargs.get("dynamical_model", None), HallucinatedModel
        )
        a_env = adversarial_to_antagonist_environment(environment, strong_antagonist)
        a_a = getattr(import_module("rllib.agent"), f"{antagonist_name}Agent").default(
            a_env, comment="Antagonist", *args, **kwargs
        )
        if hasattr(a_a, "dynamical_model"):
            a_a.dynamical_model.dim_action = a_env.antagonist_dim_action
        return a_a
