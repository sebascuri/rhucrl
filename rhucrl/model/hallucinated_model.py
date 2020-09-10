"""Implementation of an Optimistic Model."""
import torch
from rllib.model.transformed_model import TransformedModel


class HallucinatedModel(TransformedModel):
    """A Hallucinated Model returns a Delta at the hallucinated next state."""

    def __init__(self, base_model, transformations, beta=1.0):
        super().__init__(base_model, transformations)
        self.beta = beta

    def forward(self, state, action, next_state=None):
        """Get Optimistic Next state."""
        dim_action, dim_state = self.dim_action[0], self.dim_state[0]
        control_action = action[..., :dim_action]

        if self.model_kind == "dynamics":
            optimism_vars = action[..., dim_action : dim_action + dim_state]
        elif self.model_kind == "rewards":
            optimism_vars = action[..., -1:]
        else:
            raise NotImplementedError(
                "Hallucinated Models can only be of dynamics or rewards."
            )
        optimism_vars = torch.clamp(optimism_vars, -1.0, 1.0)

        mean, tril = self.predict(state, control_action)
        if torch.all(tril == 0.0) or optimism_vars.shape[-1] == 0:
            return mean, tril
        return (
            mean + self.beta * (tril @ optimism_vars.unsqueeze(-1)).squeeze(-1),
            torch.zeros_like(tril),
        )

    def scale(self, state, action):
        """Get scale at current state-action pair."""
        control_action = action[..., : self.dim_action[0]]
        scale = super().scale(state, control_action)

        return scale
