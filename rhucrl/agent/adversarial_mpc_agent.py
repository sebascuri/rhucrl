"""Python Script Template."""
from typing import Any, Type, TypeVar

from rllib.agent import MPCAgent, RandomAgent
from rllib.algorithms.mpc.abstract_solver import MPCSolver
from rllib.dataset.datatypes import Observation

from rhucrl.algorithm.adversarial_mpc import adversarial_solver
from rhucrl.environment.adversarial_environment import AdversarialEnv

from .adversarial_agent import AdversarialAgent

T = TypeVar("T", bound="AdversarialMPCAgent")


class AdversarialMPCAgent(AdversarialAgent):
    """AdversarialMPC Agent uses an Adversarial MPC shooting algorithm."""

    def __init__(self, mpc_solver: MPCSolver, *args: Any, **kwargs: Any):
        protagonist = MPCAgent(mpc_solver, *args, **kwargs)
        antagonist = RandomAgent(dim_state=(), dim_action=(), num_actions=1)
        super().__init__(protagonist, antagonist, *args, **kwargs)
        self.policy = protagonist.policy

    def observe(self, observation: Observation) -> None:
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
        cls: Type[T], environment: AdversarialEnv, *args: Any, **kwargs: Any
    ) -> T:
        """Get default RARL agent."""
        agent = MPCAgent.default(environment, *args, **kwargs)
        mpc_solver = adversarial_solver(
            base_solver=agent.planning_algorithm,
            protagonist_dim_action=environment.protagonist_dim_action,
            antagonist_dim_action=environment.antagonist_dim_action,
        )
        return super().default(environment, mpc_solver=mpc_solver, *args, **kwargs)
