"""Zero sum agent."""
from importlib import import_module
from itertools import product

from rllib.dataset.datatypes import Observation
from rllib.model import TransformedModel

from rhucrl.model import HallucinatedModel
from rhucrl.policy.split_policy import SplitPolicy

from .adversarial_agent import AdversarialAgent


class ZeroSumAgent(AdversarialAgent):
    """Zero-Sum Agent.

    Zero-Sum has two dependent agents.
    The protagonist receives (s, a, r, s') and the protagonist (s, a, -r, s').
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for agent_a, agent_b in product(self.agents.values(), self.agents.values()):
            assert agent_a.policy.base_policy is agent_b.policy.base_policy

        self.policy = self.agents["Antagonist"].policy

    def observe(self, observation):
        """Send observations to both players.

        The protagonist receives (s, a, r, s') and the antagonist (s, a, -r, s').

        """
        super().observe(observation)
        protagonist_observation = Observation(*observation)
        antagonist_observation = Observation(*observation)
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
        """Get default Zero-Sum agent."""
        p_agent = ZeroSumAgent.get_default_protagonist(
            environment, dynamical_model=protagonist_dynamical_model, *args, **kwargs
        )
        sa_agent = ZeroSumAgent.get_default_antagonist(
            environment, dynamical_model=antagonist_dynamical_model, *args, **kwargs
        )
        sa_agent.policy.base_policy = p_agent.policy.base_policy
        sa_agent.set_policy(sa_agent.policy)
        sa_agent.model_learning_algorithm = None

        if isinstance(antagonist_dynamical_model, HallucinatedModel):
            expected_model = TransformedModel(
                antagonist_dynamical_model.base_model,
                antagonist_dynamical_model.forward_transformations,
            )
            wa_agent = ZeroSumAgent.get_default_antagonist(
                environment, dynamical_model=expected_model, *args, **kwargs
            )
            wa_agent.policy.base_policy = p_agent.policy.base_policy
            wa_agent.set_policy(wa_agent.policy)
            wa_agent.model_learning_algorithm = None
        else:
            wa_agent = None

        return super().default(
            environment,
            protagonist_agent=p_agent,
            antagonist_agent=sa_agent,
            weak_antagonist_agent=wa_agent,
            *args,
            **kwargs,
        )

    @staticmethod
    def get_default_protagonist(environment, protagonist_name="SAC", *args, **kwargs):
        """Get protagonist using RARL."""
        agent_ = getattr(import_module("rllib.agent"), f"{protagonist_name}Agent")
        protagonist_agent = agent_.default(
            environment, comment="Protagonist", *args, **kwargs
        )
        protagonist_policy = SplitPolicy(
            base_policy=protagonist_agent.policy,
            protagonist_dim_action=environment.protagonist_dim_action,
            antagonist_dim_action=environment.antagonist_dim_action,
            protagonist=True,
        )
        protagonist_agent.set_policy(protagonist_policy)
        return protagonist_agent

    @staticmethod
    def get_default_antagonist(environment, antagonist_name="SAC", *args, **kwargs):
        """Get protagonist using RARL."""
        strong_antagonist = isinstance(
            kwargs.get("dynamical_model", None), HallucinatedModel
        )
        antagonist_agent = getattr(
            import_module("rllib.agent"), f"{antagonist_name}Agent"
        ).default(
            environment,
            comment=f"{'Strong' if strong_antagonist else 'Weak'} Antagonist",
            *args,
            **kwargs,
        )
        antagonist_policy = SplitPolicy(
            base_policy=antagonist_agent.policy,
            protagonist_dim_action=environment.protagonist_dim_action,
            antagonist_dim_action=environment.antagonist_dim_action,
            protagonist=False,
            weak_antagonist=not strong_antagonist,
            strong_antagonist=strong_antagonist,
        )
        antagonist_agent.set_policy(antagonist_policy)
        return antagonist_agent
