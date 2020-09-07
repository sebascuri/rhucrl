"""Implementation of an Optimistic Model."""
import torch
from rllib.model.transformed_model import TransformedModel


class HallucinatedModel(TransformedModel):
    """A Hallucinated Model returns a Delta at the hallucinated next state."""

    def __init__(
        self, base_model, transformations, beta=1.0, hallucinate_rewards=False
    ):
        super().__init__(base_model, transformations)
        self._true_dim_action = base_model.dim_action
        if hallucinate_rewards:
            self.dim_action = (self.dim_action[0] + self.dim_state[0] + 1,)
        else:
            self.dim_action = (self.dim_action[0] + self.dim_state[0],)
        self.beta = beta

    def forward(self, state, action, next_state=None):
        """Get Optimistic Next state."""
        true_dim_action, dim_state = self._true_dim_action[0], self.dim_state[0]
        control_action = action[..., :true_dim_action]

        if self.model_kind == "dynamics":
            optimism_vars = action[..., true_dim_action : true_dim_action + dim_state]
        elif self.model_kind == "rewards":
            optimism_vars = action[..., -1:]
        else:
            raise NotImplementedError(
                "Hallucinated Models can only be of dynamics or rewards."
            )
        optimism_vars = torch.clamp(optimism_vars, -1.0, 1.0)

        mean, tril = self.predict(state, control_action)
        if torch.all(tril == 0.0):
            return mean

        return (
            mean + self.beta * (tril @ optimism_vars.unsqueeze(-1)).squeeze(-1),
            torch.zeros_like(tril),
        )

    def scale(self, state, action):
        """Get scale at current state-action pair."""
        control_action = action[..., : self._true_dim_action[0]]
        scale = super().scale(state, control_action)

        return scale
