"""Python Script Template."""
from importlib import import_module

import torch
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
            dim_action=dim_action,
            action_scale=action_scale,
            protagonist_policy=self.protagonist.policy,
            antagonist_policy=self.antagonist.policy,
        )
        self.hallucinate = hallucinate

    def observe(self, observation) -> None:
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

        p_dim = self.policy.protagonist_dim_action[0] - h_dim
        a_dim = self.policy.antagonist_dim_action[0] - h_dim

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
        self.policy.protagonist_policy = self.protagonist.policy
        self.policy.antagonist_policy = self.antagonist.policy

    def load_protagonist(self, path, idx=0):
        """Load protagonist and copy policy to joint policy."""
        super().load_protagonist(path, idx)
        self.policy.protagonist_policy = self.protagonist.policy

    def load_antagonist(self, path, idx=0):
        """Load protagonist and copy policy to joint policy."""
        super().load_antagonist(path, idx)
        self.policy.antagonist_policy = self.antagonist.policy

    @classmethod
    def default(cls, environment, base_agent_name="PPO", *args, **kwargs):
        """Get default RARL agent."""
        agent_module = import_module("rllib.agent")
        agent_ = getattr(agent_module, f"{base_agent_name}Agent")

        p_env = adversarial_to_protagonist_environment(environment)
        p_agent = agent_.default(p_env, *args, **kwargs)

        a_env = adversarial_to_antagonist_environment(environment)
        a_agent = agent_.default(a_env, *args, **kwargs)

        return super().default(
            environment,
            dim_action=environment.dim_action,
            action_scale=environment.action_scale,
            protagonist_agent=p_agent,
            antagonist_agent=a_agent,
            *args,
            **kwargs,
        )
