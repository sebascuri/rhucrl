"""Python Script Template."""

from .adversarial_policy import AdversarialPolicy


class JointPolicy(AdversarialPolicy):
    """Given a protagonist and an antagonist policy, combine to give a joint policy."""

    def __init__(self, dim_action, action_scale, *args, **kwargs) -> None:
        super().__init__(
            dim_action=dim_action, action_scale=action_scale, *args, **kwargs
        )

    def forward(self, state):
        """Forward compute the policy."""
        p_dim = self.protagonist_policy.dim_action[0]
        a_dim = self.antagonist_policy.dim_action[0]

        p_mean, p_scale_tril = self.protagonist_policy(state)
        a_mean, a_scale_tril = self.antagonist_policy(state)

        p_std = p_scale_tril.diagonal(dim1=-1, dim2=-2)
        a_std = a_scale_tril.diagonal(dim1=-1, dim2=-2)

        if self.protagonist:
            h_mean = p_mean[..., p_dim:]
            h_std = p_std[..., p_dim:]
        elif self.antagonist:
            h_mean = a_mean[..., a_dim:]
            h_std = a_std[..., a_dim:]
        else:
            raise NotImplementedError
        if p_dim + a_dim < self.dim_action[0]:
            h_mean, h_scale_tril = self.hallucination_policy(state)
            h_std = h_scale_tril.diagonal(dim1=-1, dim2=-2)

        p_mean, p_std = p_mean[..., :p_dim], p_std[..., :p_dim]
        a_mean, a_std = a_mean[..., :a_dim], a_std[..., :a_dim]

        return self.stack_policies((p_mean, a_mean, h_mean), (p_std, a_std, h_std))
