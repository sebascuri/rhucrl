"""Python Script Template."""

from rllib.agent import MPCAgent, RandomAgent
from rllib.dataset.datatypes import Observation

from rhucrl.algorithm.adversarial_mpc import adversarial_solver

from .adversarial_agent import AdversarialAgent


class AdversarialMPCAgent(AdversarialAgent):
    """AdversarialMPC Agent uses an Adversarial MPC shooting algorithm."""

    def __init__(self, mpc_solver, *args, **kwargs):
        protagonist = MPCAgent(mpc_solver, *args, **kwargs)
        antagonist = RandomAgent(dim_state=(), dim_action=(), num_actions=1)
        super().__init__(protagonist, antagonist, *args, **kwargs)
        self.policy = protagonist.policy

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
    def default(cls, environment, *args, **kwargs):
        """Get default RARL agent."""
        agent = MPCAgent.default(environment, *args, **kwargs)
        mpc_solver = adversarial_solver(
            base_solver=agent.planning_algorithm,
            protagonist_dim_action=environment.protagonist_dim_action,
            antagonist_dim_action=environment.antagonist_dim_action,
        )
        return super().default(environment, mpc_solver=mpc_solver, *args, **kwargs)
