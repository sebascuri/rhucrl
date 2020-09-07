"""Python Script Template."""
from typing import Tuple

import torch
from rllib.algorithms.mpc.abstract_solver import MPCSolver
from rllib.util.neural_networks.utilities import repeat_along_dimension


def adversarial_solver(
    base_solver: MPCSolver,
    protagonist_dim_action: Tuple[int],
    antagonist_dim_action: Tuple[int],
) -> MPCSolver:
    """Get Adversarial MPC Shooting algorithm class."""
    #

    class AdversarialMPCShooting(base_solver.__class__):  # type: ignore
        """Adversarial MPC Shooting algorithm."""

        def __init__(
            self,
            base_solver: MPCSolver,
            protagonist_dim_action: Tuple[int],
            antagonist_dim_action: Tuple[int],
        ):
            super().__init__(
                **{**base_solver.__dict__, **dict(base_solver.named_modules())}
            )
            self.p_dim_action = protagonist_dim_action
            self.a_dim_action = antagonist_dim_action
            self.h_dim_aciton = (
                self.dynamical_model.dim_action[0]
                - protagonist_dim_action[0]
                - antagonist_dim_action[0],
            )

        def forward(self, state):
            """Return action that solves the MPC problem."""
            self.dynamical_model.eval()
            batch_shape = state.shape[:-1]
            self.initialize_actions(batch_shape)

            state = repeat_along_dimension(state, number=self.num_samples, dim=-2)

            # max_{protagonist} min_{antagonist} max_{hallucination}
            for _ in range(self.num_iter):
                action_sequence = self.get_candidate_action_sequence()
                returns = self.evaluate_action_sequence(action_sequence, state)

                max_actions = self.get_best_action(action_sequence, returns)
                min_actions = self.get_best_action(action_sequence, -returns)

                # Be optimistic about the model by maximizing h_action.
                p_action, _, h_action = max_actions.split(
                    [self.p_dim_action[0], self.a_dim_action[0], self.h_dim_aciton[0]],
                    -1,
                )
                _, a_action, _ = min_actions.split(
                    [self.p_dim_action[0], self.a_dim_action[0], self.h_dim_aciton[0]],
                    -1,
                )

                elite_actions = torch.cat([p_action, a_action, h_action], dim=-1)

                self.update_sequence_generation(elite_actions)

            p_action = repeat_along_dimension(
                self.mean[..., : self.p_dim_action[0]], number=self.num_samples, dim=-2
            )
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

    return AdversarialMPCShooting(
        base_solver=base_solver,
        protagonist_dim_action=protagonist_dim_action,
        antagonist_dim_action=antagonist_dim_action,
    )
