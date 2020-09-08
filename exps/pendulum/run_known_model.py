"""Python Script Template."""

from rllib.model import TransformedModel
from rllib.util.utilities import set_random_seed

from exps.pendulum.utilities import (
    PendulumModel,
    PendulumReward,
    pendulum_reset,
    pendulum_reward,
)
from exps.utilities import get_agent, get_command_line_parser, get_environment
from rhucrl.environment.wrappers import ResetWrapper, RewardWrapper
from rhucrl.utilities.training import train_adversarial_agent

parser = get_command_line_parser()
parser.set_defaults(
    environment="PendulumAdvEnv-v0",
    agent="AdversarialMPC",
    attack_mode="gravity",
    alpha=0.1,
)
args = parser.parse_args()

set_random_seed(args.seed)
# %% Generate environment.
environment = get_environment(args, attack_mode="gravity")
environment.add_wrapper(RewardWrapper, reward_function=pendulum_reward)
environment.add_wrapper(ResetWrapper, reset_function=pendulum_reset)

# %% Generate Agent.
agent = get_agent(
    args,
    environment,
    dynamical_model=TransformedModel(
        PendulumModel(alpha=args.alpha, attack_mode=args.attack_mode), []
    ),
    reward_model=PendulumReward(),
)

# %% Train Agent.
train_adversarial_agent(
    mode="both",
    agent=agent,
    environment=environment,
    num_episodes=10,
    max_steps=200,
    print_frequency=1,
    eval_frequency=0,
    render=True,
)
