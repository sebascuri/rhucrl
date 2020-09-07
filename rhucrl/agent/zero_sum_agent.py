"""Python Script Template."""
from importlib import import_module

from rllib.dataset.datatypes import Observation
from rllib.util.neural_networks.utilities import deep_copy_module

from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.policy.split_policy import SplitPolicy

from .adversarial_agent import AdversarialAgent


class ZeroSumAgent(AdversarialAgent):
    """Zero-Sum Agent.

    Zero-Sum has two dependent agents.
    The protagonist receives (s, a, r, s') and the adversary (s, a, -r, s').

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.protagonist_agent.policy.base_policy
            is self.adversarial_agent.policy.base_policy
        ), "Protagonist and Adversarial agent should share the base policy."
        self.policy = self.protagonist_agent.policy

    def observe(self, observation: Observation) -> None:
        """Send observations to both players.

        This is the crucial method as it needs to separate the actions.
        """
        super().observe(observation)
        protagonist_observation = Observation(*observation)
        adversarial_observation = Observation(*observation)
        adversarial_observation.reward = -observation.reward

        self.send_observations(protagonist_observation, adversarial_observation)

    @classmethod
    def default(
        cls, environment: AdversarialEnv, base_agent_name: str = "SAC", *args, **kwargs
    ):
        """Get default Zero-Sum agent."""
        agent = getattr(import_module("rllib.agent"), f"{base_agent_name}Agent")
        protagonist_agent = agent.default(environment, *args, **kwargs)
        adversarial_agent = agent.default(environment, *args, **kwargs)

        protagonist_policy = SplitPolicy(
            base_policy=protagonist_agent.policy,
            protagonist_dim_action=environment.protagonist_dim_action,
            adversarial_dim_action=environment.adversarial_dim_action,
            protagonist=True,
        )

        adversarial_policy = SplitPolicy(
            base_policy=protagonist_agent.policy,
            protagonist_dim_action=environment.protagonist_dim_action,
            adversarial_dim_action=environment.adversarial_dim_action,
            protagonist=False,
        )

        for agent, policy in zip(
            (protagonist_agent, adversarial_agent),
            (protagonist_policy, adversarial_policy),
        ):
            agent.policy = policy
            agent.algorithm.policy = policy
            agent.algorithm.policy_target = deep_copy_module(policy)

        return cls(
            protagonist_agent,
            adversarial_agent,
            train_frequency=protagonist_agent.train_frequency,
            num_iter=protagonist_agent.num_iter,
            num_rollouts=protagonist_agent.num_rollouts,
            *args,
            **kwargs,
        )
