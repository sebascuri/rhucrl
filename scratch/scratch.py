"""Python Script Template."""
from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.environment.adversarial_wrapper import (
    NoisyActionRobustWrapper,
    ProbabilisticActionRobustWrapper,
)

from rhucrl.environment import ENVIRONMENTS


for environment in ENVIRONMENTS:
    env = AdversarialEnv(environment)
    env.reset()
    env.step(env.env.unwrapped.action_space.sample())
    env.step(env.action_space.sample())
    assert env.alpha == 5.0

for wrapper in [NoisyActionRobustWrapper, ProbabilisticActionRobustWrapper]:
    for environment in ["HalfCheetah-v2", "Pendulum-v0"]:
        print(environment, wrapper)
        env = AdversarialEnv(environment)
        print(env.adversarial_action_dim, env.original_action_dim)
        env.alpha = 1.0
        assert env.alpha == 0

        env.reset()
        env.step(env.env.unwrapped.action_space.sample())
        env.step(env.action_space.sample())

        env.add_wrapper(wrapper, alpha=0.4)
        assert env.alpha == 0.4
        print(env.adversarial_action_dim, env.original_action_dim, env.env.alpha)
        env.alpha = 1.0
        assert env.alpha == 1.0
        env.reset()
        env.step(env.env.unwrapped.action_space.sample())
        env.step(env.action_space.sample())
