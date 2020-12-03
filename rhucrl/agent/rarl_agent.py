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

    def __init__(self, dim_action, action_scale, hallucinate=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = JointPolicy(
            dim_state=self.protagonist.policy.dim_state,
            dim_action=dim_action,
            action_scale=action_scale,
            protagonist_policy=self.protagonist.policy,
            antagonist_policy=self.antagonist.policy,
        )
        self.hallucinate = hallucinate

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

        if self.hallucinate:
            h_dim = self.policy.dim_state[0]
        else:
            h_dim = 0

        p_dim = self.policy.protagonist_policy.dim_action[0] - h_dim
        a_dim = self.policy.protagonist_policy.dim_action[0] - h_dim

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

    @classmethod
    def default(
        cls,
        environment,
        base_agent_name="PPO",
        hallucinate_protagonist=False,
        hallucinate_antagonist=False,
        *args,
        **kwargs,
    ):
        """Initialize RARL by default."""
        agent_module = import_module("rllib.agent")
        agent_ = getattr(agent_module, f"{base_agent_name}Agent")

        p_env = adversarial_to_protagonist_environment(environment=environment)
        if hallucinate_protagonist:
            policy = AugmentedPolicy.default(p_env, *args, **kwargs)
            p_env.add_wrapper(HallucinationWrapper)
        else:
            policy = NNPolicy.default(p_env, *args, **kwargs)
        p_agent = agent_.default(p_env, policy=policy, *args, **kwargs)

        a_env = adversarial_to_antagonist_environment(environment=environment)
        if hallucinate_antagonist:
            policy = AugmentedPolicy.default(a_env, *args, **kwargs)
            a_env.add_wrapper(HallucinationWrapper)
        else:
            policy = NNPolicy.default(a_env, *args, **kwargs)
        a_agent = agent_.default(a_env, policy=policy, *args, **kwargs)

        return super().default(
            environment,
            dim_action=environment.dim_action,
            action_scale=environment.action_scale,
            protagonist_agent=p_agent,
            antagonist_agent=a_agent,
            hallucinate=hallucinate_protagonist or hallucinate_antagonist,
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
