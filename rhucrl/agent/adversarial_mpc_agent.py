"""Python Script Template."""
from contextlib import nullcontext

from rllib.agent import MPCAgent
from rllib.algorithms.mpc import CEMShooting

from rhucrl.algorithm.adversarial_mpc import adversarial_solver
from rhucrl.environment.utilities import Hallucinate
from rhucrl.utilities.util import get_default_models

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
        reward_model=None,
        termination_model=None,
        hallucinate=False,
        strong_antagonist=False,
        nominal_model=False,
        *args,
        **kwargs,
    ):
        """Get default RARL agent."""
        dynamical_model, reward_model, termination_model = get_default_models(
            environment,
            known_dynamical_model=dynamical_model,
            known_reward_model=reward_model,
            known_termination_model=termination_model,
            hallucinate=hallucinate,
        )
        if hallucinate:
            cm = Hallucinate(environment)
        else:
            cm = nullcontext()

        with cm:
            base_solver = CEMShooting(
                dynamical_model=dynamical_model,
                reward_model=reward_model,
                termination_model=termination_model,
                action_scale=environment.action_scale,
                *args,
                **kwargs,
            )

            mpc_solver = adversarial_solver(
                base_solver=base_solver,
                protagonist_dim_action=environment.protagonist_dim_action,
                antagonist_dim_action=environment.antagonist_dim_action,
                strong_antagonist=strong_antagonist,
                nominal_model=nominal_model,
            )
            return super().default(environment, mpc_solver=mpc_solver, *args, **kwargs)
