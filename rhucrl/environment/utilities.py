"""Python Script Template."""
from gym.spaces import Box

from .adversarial_environment import AdversarialEnv


def adversarial_to_protagonist_environment(
    environment: AdversarialEnv,
) -> AdversarialEnv:
    """Get the environment as seen by the protagonist."""
    protagonist_environment = AdversarialEnv(environment.env_name)
    protagonist_dim_action = environment.protagonist_dim_action[0]

    protagonist_environment.dim_action = (protagonist_dim_action,)
    protagonist_environment.action_space = Box(
        environment.action_space.low[:protagonist_dim_action],
        environment.action_space.high[:protagonist_dim_action],
    )
    return protagonist_environment


def adversarial_to_adversary_environment(
    environment: AdversarialEnv,
) -> AdversarialEnv:
    """Get the environment as seen by the adversary."""
    adversary_environment = AdversarialEnv(environment.env_name)
    protagonist_dim_action = environment.protagonist_dim_action[0]
    adversary_environment.dim_action = (
        environment.dim_action[0] - protagonist_dim_action,
    )
    adversary_environment.action_space = Box(
        environment.action_space.low[protagonist_dim_action:],
        environment.action_space.high[protagonist_dim_action:],
    )
    return adversary_environment
