"""Python Script Template."""
from contextlib import nullcontext

from rllib.agent import MPCAgent

from rhucrl.algorithm.adversarial_mpc import adversarial_solver
from rhucrl.environment.utilities import Hallucinate
from rhucrl.utilities.util import get_default_model

from .adversarial_agent import AdversarialAgent


class AdversarialMPCAgent(AdversarialAgent):
    """AdversarialMPC Agent uses an Adversarial MPC shooting algorithm."""

    def __init__(self, mpc_solver, *args, **kwargs):
        protagonist = MPCAgent(mpc_solver, *args, **kwargs)
        super().__init__(protagonist, *args, **kwargs)
        self.policy = protagonist.policy
        self.policy.dim_action = (
            mpc_solver.p_dim_action[0]
            + mpc_solver.a_dim_action[0]
            + mpc_solver.h_dim_action[0],
        )

    def observe(self, observation):
        """Send observations to both players.

        The protagonist receives (s, a, r, s') and the antagonist (s, a, -r, s').

        """
        super().observe(observation)
        protagonist_observation = observation.clone()
        antagonist_observation = observation.clone()
        antagonist_observation.reward = -observation.reward

        self.send_observations(protagonist_observation, antagonist_observation)

    @classmethod
    def default(
        cls,
        environment,
        dynamical_model=None,
        hallucinate=False,
        strong_antagonist=False,
        *args,
        **kwargs,
    ):
        """Get default RARL agent."""
        dynamical_model = get_default_model(
            environment, known_model=dynamical_model, hallucinate=hallucinate
        )
        if hallucinate:
            cm = Hallucinate(environment)
        else:
            cm = nullcontext()

        with cm:
            agent = MPCAgent.default(
                environment, dynamical_model=dynamical_model, *args, **kwargs
            )
            mpc_solver = adversarial_solver(
                base_solver=agent.planning_algorithm,
                protagonist_dim_action=environment.protagonist_dim_action,
                antagonist_dim_action=environment.antagonist_dim_action,
                strong_antagonist=strong_antagonist,
            )
            return super().default(environment, mpc_solver=mpc_solver, *args, **kwargs)
