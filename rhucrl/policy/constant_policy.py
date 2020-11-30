"""Python Script Template."""
import torch
from rllib.policy import AbstractPolicy


class ConstantPolicy(AbstractPolicy):
    """A constant policy is independent of the state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.discrete_action:
            self.logits = torch.rand(self.num_actions)
            self.logits.requires_grad = True
        else:
            self.mean = torch.randn(self.dim_action)
            self.std = torch.randn(self.dim_action)
            self.mean.requires_grad = True
            self.std.requires_grad = True

    def forward(self, state):
        """Compute action distribution."""
        if self.discrete_state:
            batch_size = state.shape
        else:
            batch_size = state.shape[:-1]

        if self.discrete_action:
            logits = self.logits.repeat(*batch_size, 1)
            return logits
        else:
            mean = self.mean.repeat(*batch_size, 1)
            std = self.std.repeat(*batch_size, 1)
            return mean, std.diag_embed()
