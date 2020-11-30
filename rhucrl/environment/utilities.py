"""Python Script Template."""
import numpy as np
from gym.spaces import Box

from rhucrl.environment.wrappers import HallucinationWrapper

from .adversarial_environment import AdversarialEnv


def adversarial_to_protagonist_environment(environment):
    """Get the environment as seen by the protagonist."""
    protagonist_environment = AdversarialEnv(environment.env_name)
    protagonist_dim_action = environment.protagonist_dim_action[0]
    antagonist_dim_action = environment.antagonist_dim_action[0]

    protagonist_environment.dim_action = (
        environment.dim_action[0] - antagonist_dim_action,
    )
    protagonist_environment.action_space = Box(
        np.concatenate(
            (
                environment.action_space.low[:protagonist_dim_action],
                environment.action_space.low[
                    protagonist_dim_action + antagonist_dim_action :
                ],
            )
        ),
        np.concatenate(
            (
                environment.action_space.high[:protagonist_dim_action],
                environment.action_space.high[
                    protagonist_dim_action + antagonist_dim_action :
                ],
            )
        ),
    )
    return protagonist_environment


def adversarial_to_antagonist_environment(environment):
    """Get the environment as seen by the antagonist."""
    antagonist_environment = AdversarialEnv(environment.env_name)
    protagonist_dim_action = environment.protagonist_dim_action[0]
    dim_action = environment.antagonist_dim_action[0]

    action_space = Box(
        environment.action_space.low[
            protagonist_dim_action : protagonist_dim_action + dim_action
        ],
        environment.action_space.high[
            protagonist_dim_action : protagonist_dim_action + dim_action
        ],
    )
    antagonist_environment.action_space = action_space
    antagonist_environment.env.action_space = action_space

    antagonist_environment.dim_action = (dim_action,)
    return antagonist_environment


class Hallucinate(object):
    """Context manager to hallucinate an environment."""

    def __init__(self, environment, hallucinate_rewards=False):
        self.environment = environment
        self.hallucinate_rewards = hallucinate_rewards

    def __enter__(self):
        """Enter into a Hallucination Context."""
        self.environment.add_wrapper(
            HallucinationWrapper, hallucinate_rewards=self.hallucinate_rewards
        )

    def __exit__(self, *args):
        """Exit the Hallucination Context."""
        self.environment.pop_wrapper()
