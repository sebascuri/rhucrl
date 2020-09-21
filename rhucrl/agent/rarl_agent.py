"""Python Script Template."""
from contextlib import nullcontext
from importlib import import_module

import torch

from rhucrl.environment.utilities import (
    Hallucinate,
    adversarial_to_antagonist_environment,
    adversarial_to_protagonist_environment,
)
from rhucrl.model.hallucinated_model import HallucinatedModel
from rhucrl.policy.joint_policy import JointPolicy
from rhucrl.utilities.util import get_default_models

from .adversarial_agent import AdversarialAgent


class RARLAgent(AdversarialAgent):
    """RARL Agent.

    RARL has two independent agents.
    The protagonist receives (s, a_pro, r, s') and the antagonist (s, a_ant, -r, s').

    """

    policy: JointPolicy

    def __init__(self, dim_action, action_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        antagonist = self.agents.get("Antagonist", None)
        if hasattr(antagonist, "policy"):
            antagonist_policy = antagonist.policy
        else:
            antagonist_policy = None
        self.policy = JointPolicy(
            dim_action=dim_action,
            action_scale=action_scale,
            protagonist_policy=self.agents["Protagonist"].policy,
            antagonist_policy=antagonist_policy,
        )
        if hasattr(antagonist, "dynamical_model"):
            self.strong_antagonist = isinstance(
                antagonist.dynamical_model, HallucinatedModel
            )
        else:
            self.strong_antagonist = False

    def observe(self, observation) -> None:
        """Send observations to both agents.

        Send to the protagonist (s, a_p, r, s', other)
        Send to antagonist (s, a_a, -r, s', other).
        """
        super().observe(observation)
        p_observation = observation.clone()
        a_observation = observation.clone()

        protagonist = self.agents["Protagonist"]
        if hasattr(protagonist, "dynamical_model") and isinstance(
            protagonist.dynamical_model, HallucinatedModel
        ):
            h_dim = self.policy.dim_state[0]
        else:
            h_dim = 0
        p_dim = self.policy.protagonist_dim_action[0] - h_dim
        if self.strong_antagonist:
            a_dim = self.policy.antagonist_dim_action[0] - h_dim
        else:
            a_dim = self.policy.antagonist_dim_action[0]
        p_observation.action = torch.cat(
            (observation.action[:p_dim], observation.action[p_dim + a_dim :]), -1
        )
        if self.strong_antagonist:
            a_observation.action = observation.action[p_dim:]
        else:
            a_observation.action = observation.action[p_dim : p_dim + a_dim]
        a_observation.reward = -observation.reward

        self.send_observations(p_observation, a_observation)

    def load_protagonist(self, path):
        """Load protagonist and copy policy to joint policy."""
        super().load_protagonist(path)
        self.policy.protagonist_policy = self.agents["Protagonist"].policy

    @classmethod
    def default(cls, environment, hallucinate=False, *args, **kwargs):
        """Get default RARL agent."""
        p_agent = RARLAgent.get_default_protagonist(
            environment, hallucinate=hallucinate, *args, **kwargs
        )
        a_agent = RARLAgent.get_default_antagonist(
            environment, hallucinate=hallucinate, *args, **kwargs
        )
        if hasattr(a_agent, "reward_model"):
            a_agent.reward_model.action_cost_ratio = 0.0

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
        reward_model=None,
        termination_model=None,
        hallucinate=False,
        strong_antagonist=False,
        *args,
        **kwargs,
    ):
        """Get protagonist using RARL."""
        p_env = adversarial_to_protagonist_environment(environment)
        dynamical_model, reward_model, termination_model = get_default_models(
            p_env,
            known_dynamical_model=dynamical_model,
            known_reward_model=reward_model,
            known_termination_model=termination_model,
            hallucinate=hallucinate,
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
                reward_model=reward_model,
                termination_model=termination_model,
                *args,
                **kwargs,
            )

    @staticmethod
    def get_default_antagonist(
        environment,
        antagonist_name="SAC",
        dynamical_model=None,
        reward_model=None,
        termination_model=None,
        hallucinate=False,
        strong_antagonist=False,
        *args,
        **kwargs,
    ):
        """Get protagonist using RARL."""
        # Only hallucinate if the agent is strong
        hallucinate = hallucinate and strong_antagonist
        if hallucinate:
            cm = Hallucinate(environment)
        else:
            cm = nullcontext()
        with cm:
            a_env = adversarial_to_antagonist_environment(environment, hallucinate)
            dynamical_model, reward_model, termination_model = get_default_models(
                a_env,
                known_dynamical_model=dynamical_model,
                known_reward_model=reward_model,
                known_termination_model=termination_model,
                hallucinate=hallucinate,
                strong_antagonist=strong_antagonist,
            )
            return getattr(
                import_module("rllib.agent"), f"{antagonist_name}Agent"
            ).default(
                a_env,
                comment="Strong Antagonist" if hallucinate else "Weak Antagonist",
                dynamical_model=dynamical_model,
                reward_model=reward_model,
                termination_model=termination_model,
                *args,
                **kwargs,
            )
