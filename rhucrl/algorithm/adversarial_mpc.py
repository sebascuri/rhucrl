"""Python Script Template."""
from contextlib import nullcontext
from typing import Any, Tuple

import torch
import torch.nn as nn
from rllib.algorithms.mpc.abstract_solver import MPCSolver
from rllib.util.neural_networks.utilities import repeat_along_dimension

from rhucrl.model import HallucinatedModel


class AdversarialMPCShooting(nn.Module):  # type: ignore
    r"""Adversarial MPC Shooting algorithm.

    Parameters
    ----------
    base_solver: MPCSolver
        initialized base solver that this class inherits from.
    protagonist_dim_action: Tuple[int].
        Dimensions of protagonist actions.
    antagonist_dim_action: Tuple[int].
        Dimension of antagonist actions.

    nominal_model: bool, optional (default=False).
        If True, the protagonist will plan with the nominal model, ie alpha=0.
        If False, the protagonist will know how the antagonist affects the system.
    """

    def __init__(
        self,
        base_solver: MPCSolver,
        protagonist_dim_action: Tuple[int],
        antagonist_dim_action: Tuple[int],
        nominal_model: bool,
    ) -> None:
        super().__init__()
        self.base_solver = base_solver
        self.p_dim_action = protagonist_dim_action
        self.a_dim_action = antagonist_dim_action
        if isinstance(base_solver.dynamical_model, HallucinatedModel):
            self.h_dim_action = base_solver.dynamical_model.dim_state
        else:
            self.h_dim_action = (0,)
        self.nominal_model = nominal_model

        self.dim_action = (
            self.p_dim_action[0] + self.a_dim_action[0] + self.h_dim_action[0]
        )
        self.base_solver.dim_action = (
            self.p_dim_action[0] + self.a_dim_action[0] + self.h_dim_action[0]
        )

    def __getattr__(self, name: str) -> Any:
        """Get attribute with given name."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_solver, name)

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
                [self.p_dim_action[0], self.a_dim_action[0], self.h_dim_action[0]], -1
            )
            _, a_action, _ = min_actions.split(
                [self.p_dim_action[0], self.a_dim_action[0], self.h_dim_action[0]], -1
            )

            elite_actions = torch.cat([p_action, a_action, h_action], dim=-1)

            self.update_sequence_generation(elite_actions)

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
