"""Python Script Template."""
from importlib import import_module

import torch
from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from hucrl.policy.augmented_policy import AugmentedPolicy
from rllib.policy import NNPolicy
from rllib.util.utilities import tensor_to_distribution

from rhucrl.agent.adversarial_agent import AdversarialAgent
from rhucrl.environment.utilities import (
    adversarial_to_antagonist_environment,
    adversarial_to_protagonist_environment,
)
from rhucrl.policy.joint_policy import JointPolicy


class RARLAgent(AdversarialAgent):
    """RARL Agent.

    RARL has two independent agents.
    The protagonist receives (s, a_pro, r, s') and the antagonist (s, a_ant, -r, s').

    """

    policy: JointPolicy

    def __init__(self, dim_action, action_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = JointPolicy(
            dim_state=self.protagonist.policy.dim_state,
            dim_action=dim_action,
            action_scale=action_scale,
            protagonist_policy=self.protagonist.policy,
            antagonist_policy=self.antagonist.policy,
        )

    def observe(self, observation):
        """Send observations to both agents.

        Send to the protagonist (s, a_p, r, s', other)
        Send to antagonist (s, a_a, -r, s', other).
        """
        super().observe(observation)
        p_observation = observation.clone()
        a_observation = observation.clone()

        fake_action = tensor_to_distribution(
            self.antagonist.policy(p_observation.state)
        ).sample()

        p_dim = self.policy.protagonist_policy.dim_action[0]
        a_dim = self.policy.antagonist_policy.dim_action[0]

        p_observation.action = torch.cat(
            (observation.action[:p_dim], observation.action[p_dim + a_dim :]), -1
        )
        a_observation.action = torch.cat(
            (observation.action[p_dim : p_dim + a_dim], fake_action[a_dim:]), -1
        )

        a_observation.reward = -observation.reward
        self.send_observations(p_observation, a_observation)

    def start_episode(self):
        """Start a new episode.

        Here a new protagonist and antagonist may be sampled, hence reset the policies.
        """
        super().start_episode()
        self.policy.set_protagonist_policy(self.protagonist.policy)
        self.policy.set_antagonist_policy(self.antagonist.policy)

    def load_protagonist(self, path, idx=0):
        """Load protagonist and copy policy to joint policy."""
        super().load_protagonist(path, idx)
        self.policy.set_protagonist_policy(self.protagonist.policy)

    def load_antagonist(self, path, idx=0):
        """Load protagonist and copy policy to joint policy."""
        super().load_antagonist(path, idx)
        self.policy.set_antagonist_policy(self.antagonist.policy)

    @staticmethod
    def _init_agent(
        environment, agent_class_, hallucinate, dynamical_model=None, *args, **kwargs
    ):
        if dynamical_model is not None:
            dynamical_model = type(dynamical_model).default(environment)

        if hallucinate:
            policy = AugmentedPolicy.default(environment, *args, **kwargs)
            environment.add_wrapper(HallucinationWrapper)
        else:
            policy = NNPolicy.default(environment, *args, **kwargs)

        return agent_class_.default(
            environment, policy=policy, dynamical_model=dynamical_model, *args, **kwargs
        )

    @classmethod
    def default(
        cls,
        environment,
        base_agent_name="PPO",
        hallucinate_protagonist=False,
        hallucinate_antagonist=False,
        dynamical_model=None,
        *args,
        **kwargs,
    ):
        """Initialize RARL by default."""
        agent_module = import_module("rllib.agent")
        agent_ = getattr(agent_module, f"{base_agent_name}Agent")

        p_env = adversarial_to_protagonist_environment(environment=environment)
        p_agent = RARLAgent._init_agent(
            environment=p_env,
            agent_class_=agent_,
            hallucinate=hallucinate_protagonist,
            *args,
            **kwargs,
        )
        a_env = adversarial_to_antagonist_environment(environment=environment)
        a_agent = RARLAgent._init_agent(
            environment=a_env,
            agent_class_=agent_,
            hallucinate=hallucinate_antagonist,
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


class HRARLAgent(RARLAgent):
    """Hallucinated-RARL algorithm."""

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """Initialize RARL with Hallucination by default."""
        return RARLAgent.default(
            environment=environment,
            hallucinate_antagonist=True,
            hallucinate_protagonist=True,
            *args,
            **kwargs,
        )


class RAPAgent(RARLAgent):
    """RAP Agent."""

    def __init__(self, n_antagonists=5, *args, **kwargs):
        super().__init__(n_protagonists=1, n_antagonists=n_antagonists, *args, **kwargs)
