"""Python Script Template."""

from .adversarial_policy import AdversarialPolicy


class JointPolicy(AdversarialPolicy):
    """Given a protagonist and an antagonist policy, combine to give a joint policy."""

    def __init__(
        self,
        dim_action,
        action_scale,
        protagonist_policy,
        antagonist_policy,
        *args,
        **kwargs,
    ) -> None:
        if antagonist_policy is None:
            super().__init__(
                *args,
                dim_state=protagonist_policy.dim_state,
                dim_action=dim_action,
                protagonist_dim_action=protagonist_policy.dim_action,
                antagonist_dim_action=(0,),
                deterministic=protagonist_policy.deterministic,
                action_scale=action_scale,
                dist_params=protagonist_policy.dist_params,
                **kwargs,
            )
            self.only_protagonist = True
        else:
            assert protagonist_policy.deterministic == antagonist_policy.deterministic
            assert protagonist_policy.dim_state == antagonist_policy.dim_state
            super().__init__(
                *args,
                dim_state=protagonist_policy.dim_state,
                dim_action=dim_action,
                protagonist_dim_action=protagonist_policy.dim_action,
                antagonist_dim_action=antagonist_policy.dim_action,
                deterministic=protagonist_policy.deterministic,
                action_scale=action_scale,
                dist_params=protagonist_policy.dist_params,
                **kwargs,
            )  # type: ignore
        self.protagonist_policy = protagonist_policy
        self.antagonist_policy = antagonist_policy

    def forward(self, state):
        """Forward compute the policy."""
        p_dim = self.dim_action[0] - self.antagonist_dim_action[0]
        a_dim = self.dim_action[0] - self.protagonist_dim_action[0]

        p_mean, p_scale_tril = self.protagonist_policy(state)
        if self.only_protagonist:
            return p_mean, p_scale_tril

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

        p_mean, p_std = p_mean[..., :p_dim], p_std[..., :p_dim]
        a_mean, a_std = a_mean[..., :a_dim], a_std[..., :a_dim]

        return self.stack_policies((p_mean, a_mean, h_mean), (p_std, a_std, h_std))
