"""Python Script Template."""
import importlib

import torch
from rllib.agent import ModelBasedAgent, MPCAgent

from rhucrl.algorithm.adversarial_mpc import AdversarialMPCShooting
from rhucrl.environment.adversarial_environment import AdversarialEnv


class AdversarialMPCAgent(MPCAgent):
    """AdversarialMPC Agent uses an Adversarial MPC shooting algorithm."""

    def __init__(
        self,
        protagonist_dim_action,
        antagonist_dim_action,
        nominal_model=False,
        action_scale=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.policy.solver = AdversarialMPCShooting(
            base_solver=self.policy.solver,
            protagonist_dim_action=protagonist_dim_action,
            antagonist_dim_action=antagonist_dim_action,
            nominal_model=nominal_model,
        )
        if action_scale is None:
            action_scale = torch.ones(self.policy.dim_action)
        self.policy.action_scale = action_scale
        self.policy.solver.base_solver.action_scale = action_scale

    @classmethod
    def default(
        cls,
        environment: AdversarialEnv,
        mpc_solver=None,
        mpc_solver_name="CEMShooting",
        *args,
        **kwargs,
    ):
        """See `AbstractAgent.default'."""
        agent = ModelBasedAgent.default(environment, *args, **kwargs)
        agent.logger.delete_directory()
        kwargs.update(
            dynamical_model=agent.dynamical_model,
            reward_model=agent.reward_model,
            termination_model=agent.termination_model,
            gamma=agent.gamma,
        )
        if mpc_solver is None:
            solver_module = importlib.import_module("rllib.algorithms.mpc")
            mpc_solver = getattr(solver_module, mpc_solver_name)(
                action_scale=environment.action_scale, *args, **kwargs
            )
        return super().default(
            environment,
            protagonist_dim_action=environment.protagonist_dim_action,
            antagonist_dim_action=environment.antagonist_dim_action,
            action_scale=torch.tensor(
                environment.action_scale, dtype=torch.get_default_dtype()
            ),
            mpc_solver=mpc_solver,
            *args,
            **kwargs,
        )
