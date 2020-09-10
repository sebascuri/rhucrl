"""Python Script Template."""
import numpy as np
from gym.spaces import Box

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


def adversarial_to_antagonist_environment(environment, strong_antagonist=True):
    """Get the environment as seen by the antagonist."""
    antagonist_environment = AdversarialEnv(environment.env_name)
    protagonist_dim_action = environment.protagonist_dim_action[0]

    if strong_antagonist:
        dim_action = environment.dim_action[0] - protagonist_dim_action
    else:
        dim_action = antagonist_environment.dim_action[0] - protagonist_dim_action

    antagonist_environment.action_space = Box(
        environment.action_space.low[
            protagonist_dim_action : protagonist_dim_action + dim_action
        ],
        environment.action_space.high[
            protagonist_dim_action : protagonist_dim_action + dim_action
        ],
    )
    antagonist_environment.dim_action = (dim_action,)
    return antagonist_environment
