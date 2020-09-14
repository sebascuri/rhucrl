"""Python Script Template."""
from contextlib import nullcontext
from importlib import import_module

import torch

from rhucrl.environment.utilities import (
    Hallucinate,
    adversarial_to_antagonist_environment,
    adversarial_to_protagonist_environment,
)
from rhucrl.policy.joint_policy import JointPolicy
from rhucrl.utilities.util import get_default_model

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
            protagonist_policy=self.agents["Protagonist"].policy,
            antagonist_policy=self.agents["Antagonist"].policy,
        )

    def observe(self, observation) -> None:
        """Send observations to both players.

        This is the crucial method as it needs to separate the actions.
        """
        super().observe(observation)
        p_observation = observation.clone()
        a_observation = observation.clone()

        dim_action = self.policy.dim_action[0]
        p_dim_action = self.policy.protagonist_dim_action[0]
        a_dim_action = self.policy.antagonist_dim_action[0]
        h_dim_action = 2 * dim_action - p_dim_action - a_dim_action

        p_observation.action = torch.cat(
            (
                observation.action[: dim_action - a_dim_action],
                observation.action[-h_dim_action:],
            ),
            -1,
        )
        a_observation.action = observation.action[-a_dim_action:]
        a_observation.reward = -observation.reward

        self.send_observations(p_observation, a_observation)

    @classmethod
    def default(cls, environment, hallucinate=False, *args, **kwargs):
        """Get default RARL agent."""
        p_agent = RARLAgent.get_default_protagonist(
            environment, hallucinate=hallucinate, *args, **kwargs
        )
        a_agent = RARLAgent.get_default_antagonist(
            environment, hallucinate=hallucinate, *args, **kwargs
        )

        if hallucinate:
            cm = Hallucinate(environment)
        else:
            cm = nullcontext()
        with cm:
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
    def get_default_protagonist(
        environment,
        protagonist_name="SAC",
        dynamical_model=None,
        hallucinate=False,
        strong_antagonist=False,
        *args,
        **kwargs,
    ):
        """Get protagonist using RARL."""
        p_env = adversarial_to_protagonist_environment(environment)
        dynamical_model = get_default_model(
            p_env, known_model=dynamical_model, hallucinate=hallucinate
        )
        if hallucinate:
            cm = Hallucinate(p_env)
        else:
            cm = nullcontext()
        with cm:
            return getattr(
                import_module("rllib.agent"), f"{protagonist_name}Agent"
            ).default(
                p_env,
                comment="Protagonist",
                dynamical_model=dynamical_model,
                *args,
                **kwargs,
            )

    @staticmethod
    def get_default_antagonist(
        environment,
        antagonist_name="SAC",
        dynamical_model=None,
        hallucinate=False,
        strong_antagonist=False,
        *args,
        **kwargs,
    ):
        """Get protagonist using RARL."""
        a_env = adversarial_to_antagonist_environment(
            environment, hallucinate and strong_antagonist
        )
        dynamical_model = get_default_model(
            a_env,
            known_model=dynamical_model,
            hallucinate=hallucinate,
            protagonist=False,
            strong_antagonist=strong_antagonist,
            weak_antagonist=not strong_antagonist,
        )
        if hallucinate and strong_antagonist:
            cm = Hallucinate(environment)
        else:
            cm = nullcontext()
        with cm:
            return getattr(
                import_module("rllib.agent"), f"{antagonist_name}Agent"
            ).default(
                a_env,
                comment="Antagonist",
                dynamical_model=dynamical_model,
                *args,
                **kwargs,
            )
