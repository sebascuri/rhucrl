"""Python Script Template."""
from contextlib import nullcontext
from typing import Tuple

import torch
from rllib.algorithms.mpc.abstract_solver import MPCSolver
from rllib.util.neural_networks.utilities import repeat_along_dimension

from rhucrl.model import HallucinatedModel


def adversarial_solver(
    base_solver: MPCSolver,
    protagonist_dim_action: Tuple[int],
    antagonist_dim_action: Tuple[int],
    strong_antagonist: bool = True,
    nominal_model: bool = False,
) -> MPCSolver:
    r"""Get Adversarial MPC Shooting algorithm class.

    Parameters
    ----------
    base_solver: MPCSolver
        initialized base solver that this class inherits from.
    protagonist_dim_action: Tuple[int].
        Dimensions of protagonist actions.
    antagonist_dim_action: Tuple[int].
        Dimension of antagonist actions.
    strong_antagonist: bool, optional (default=True).
        If True, it will optimize the antagonist actions while keeping the protagonist
        fixed. i.e. \argmin_{u_antagonist} J_p(u_protagonist, u).
        ..math u_{protagonist} = \arg \max_{u_p} \min{u_a} J_o(u_p, u_a).
        ..math u_{antagonist} = \arg \min_u J_p(u_{protagonist}, u_a).

        If False, it will return the antagonist actions found by the protagonist
        training. i.e.
        ..math u_{protagonist}, u_{antagonist} = \arg \max_{u_p} \min{u_a} J_o(u_p, u_a)

    nominal_model: bool, optional (default=False).
        If True, the protagonist will plan with the nominal model, ie alpha=0.
        If False, the protagonist will know how the antagonist affects the system.
    """
    #

    class AdversarialMPCShooting(base_solver.__class__):  # type: ignore
        """Adversarial MPC Shooting algorithm."""

        def __init__(self,):
            super().__init__(
                **{**base_solver.__dict__, **dict(base_solver.named_modules())}
            )
            self.p_dim_action = protagonist_dim_action
            self.a_dim_action = antagonist_dim_action
            if isinstance(self.dynamical_model, HallucinatedModel):
                self.h_dim_action = self.dynamical_model.dim_state
            else:
                self.h_dim_action = (0,)
            self.strong_antagonist = strong_antagonist
            self.nominal_model = nominal_model

            self.dim_action = (
                self.p_dim_action[0] + self.a_dim_action[0] + self.h_dim_action[0]
            )

            self.mean = None
            self.covariance = (self._scale ** 2) * torch.eye(self.dim_action).repeat(
                self.horizon, 1, 1
            )

        def forward(self, state):
            """Return action that solves the MPC problem."""
            self.dynamical_model.eval()
            batch_shape = state.shape[:-1]
            self.initialize_actions(batch_shape)

            state = repeat_along_dimension(state, number=self.num_samples, dim=-2)

            if self.nominal_model:
                cm = NominalModel(model=self.dynamical_model.base_model)
            else:
                cm = nullcontext()
            # max_{protagonist} min_{antagonist} max_{hallucination}
            for _ in range(self.num_iter):
                action_sequence = self.get_candidate_action_sequence()
                with cm:  # Evaluate possibly with nominal model.
                    returns = self.evaluate_action_sequence(action_sequence, state)

                max_actions = self.get_best_action(action_sequence, returns)
                min_actions = self.get_best_action(action_sequence, -returns)

                # Be optimistic about the model by maximizing h_action.
                p_action, _, h_action = max_actions.split(
                    [self.p_dim_action[0], self.a_dim_action[0], self.h_dim_action[0]],
                    -1,
                )
                _, a_action, _ = min_actions.split(
                    [self.p_dim_action[0], self.a_dim_action[0], self.h_dim_action[0]],
                    -1,
                )

                elite_actions = torch.cat([p_action, a_action, h_action], dim=-1)

                self.update_sequence_generation(elite_actions)

            if not self.strong_antagonist:  # Early stop weak antagonists.
                if self.clamp:
                    return self.mean.clamp(-1.0, 1.0)
                return self.mean
            p_action = repeat_along_dimension(
                self.mean[..., : self.p_dim_action[0]], number=self.num_samples, dim=-2
            )
            # min_{antagonist} min_{hallucination}
            for _ in range(self.num_iter):
                action_sequence = self.get_candidate_action_sequence()

                # Fix protagonists actions.
                action_sequence[..., : self.p_dim_action[0]] = p_action

                returns = self.evaluate_action_sequence(action_sequence, state)
                elite_actions = self.get_best_action(action_sequence, -returns)
                self.update_sequence_generation(elite_actions)

            if self.clamp:
                return self.mean.clamp(-1.0, 1.0)
            return self.mean

    return AdversarialMPCShooting()


class NominalModel(object):
    """Given a model, make it nominal by setting alpha to zero."""

    def __init__(self, model):
        self.model = model
        self.alpha = model.alpha

    def __enter__(self):
        """Set the model alpha to zero."""
        self.model.alpha = 0.0

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Set the model alpha to previous value."""
        self.model.alpha = self.alpha
