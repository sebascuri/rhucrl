"""Python Script Template."""
from rhucrl.algorithm.maximin_algorithm import MaxiMinAlgorithm
from rhucrl.policy.split_policy import SplitPolicy


def get_maximin_agent_class(base_agent_class):
    """Get maximin agent class."""
    #

    class MaxiMinAgent(base_agent_class):
        """A maximin agent has an algorithm that optimizes a max-min problem."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.algorithm = MaxiMinAlgorithm(base_algorithm=self.algorithm)

        @classmethod
        def default(cls, environment, hallucinate_protagonist=True, *args, **kwargs):
            """See `AbstractAgent.default' method."""
            policy = SplitPolicy.default(
                environment,
                hallucinate_protagonist=hallucinate_protagonist,
                *args,
                **kwargs
            )
            return super().default(
                environment=environment, policy=policy, *args, **kwargs
            )

    return MaxiMinAgent
