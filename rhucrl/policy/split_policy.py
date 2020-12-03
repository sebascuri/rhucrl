"""Python Script Template."""

from rllib.policy import NNPolicy

from .adversarial_policy import AdversarialPolicy


class SplitPolicy(AdversarialPolicy):
    """Split a policy into protagonist and antagonist policies."""

    def __init__(self, hallucinate_protagonist=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hallucinate_protagonist = hallucinate_protagonist

    @property
    def hallucinate_protagonist(self):
        """Return true if protagonist hallucinates."""
        return self._hallucinate_protagonist

    @hallucinate_protagonist.setter
    def hallucinate_protagonist(self, value):
        """Return true if protagonist hallucinates."""
        self._hallucinate_protagonist = value

    @property
    def hallucinate_antagonist(self):
        """Return true if antagonist hallucinates."""
        return not self._hallucinate_protagonist

    @hallucinate_antagonist.setter
    def hallucinate_antagonist(self, value):
        """Return true if protagonist hallucinates."""
        self._hallucinate_protagonist = not value

    def forward(self, state):
        """Forward compute the policy."""
        p_mean, p_scale_tril = self.protagonist_policy(state)
        a_mean, a_scale_tril = self.antagonist_policy(state)
        h_mean, h_scale_tril = self.hallucination_policy(state)

        p_std = p_scale_tril.diagonal(dim1=-1, dim2=-2)
        a_std = a_scale_tril.diagonal(dim1=-1, dim2=-2)
        h_std = h_scale_tril.diagonal(dim1=-1, dim2=-2)

        if self.protagonist:
            a_mean, a_std = a_mean.detach(), a_std.detach()
            if self.hallucinate_antagonist:
                h_mean, h_std = h_mean.detach(), h_std.detach()
        elif self.antagonist:
            p_mean, p_std = p_mean.detach(), p_std.detach()
            if self.hallucinate_protagonist:
                h_mean, h_std = h_mean.detach(), h_std.detach()
        else:
            raise NotImplementedError

        return self.stack_policies((p_mean, a_mean, h_mean), (p_std, a_std, h_std))

    @classmethod
    def default(cls, environment, hallucinate_protagonist=True, *args, **kwargs):
        """See `NNPolicy.default'."""
        protagonist_policy = NNPolicy(
            dim_state=environment.dim_state,
            dim_action=environment.protagonist_dim_action,
        )
        antagonist_policy = NNPolicy(
            dim_state=environment.dim_state,
            dim_action=environment.antagonist_dim_action,
        )
        hallucination_policy = NNPolicy(
            dim_state=environment.dim_state, dim_action=environment.dim_state
        )
        return cls(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            protagonist_policy=protagonist_policy,
            antagonist_policy=antagonist_policy,
            hallucination_policy=hallucination_policy,
            hallucinate_protagonist=hallucinate_protagonist,
        )

    @classmethod
    def from_adversarial_policy(cls, adversarial_policy, *args, **kwargs):
        """Create class from an adversarial policy."""
        return cls(
            protagonist_policy=adversarial_policy.protagonist_policy,
            antagonist_policy=adversarial_policy.antagonist_policy,
            *args,
            **kwargs,
        )
