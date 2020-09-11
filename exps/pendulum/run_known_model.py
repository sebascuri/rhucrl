"""Python Script Template."""

from rllib.util.utilities import set_random_seed

from exps.pendulum.utilities import (
    AdversarialPendulumWrapper,
    PendulumModel,
    PendulumReward,
    pendulum_reset,
)
from exps.utilities import get_command_line_parser
from rhucrl.environment.wrappers import (
    NoisyActionRobustWrapper,
    ProbabilisticActionRobustWrapper,
    ResetWrapper,
)
from rhucrl.utilities.training import train_adversarial_agent
from rhucrl.utilities.util import get_agent, get_default_models, get_environment

parser = get_command_line_parser()
parser.set_defaults(
    environment="Pendulum-v1", agent="AdversarialMPC", attack_mode="gravity", alpha=0.1
)
args = parser.parse_args()

set_random_seed(args.seed)
# %% Generate environment.
environment = get_environment(args.environment, seed=args.seed)
environment.add_wrapper(ResetWrapper, reset_function=pendulum_reset)
environment.add_wrapper(
    AdversarialPendulumWrapper, alpha=args.alpha, attack_mode=args.attack_mode
)

# %% Generate Models.
protagonist_dynamical_model, antagonist_dynamical_model = get_default_models(
    environment,
    PendulumModel(alpha=args.alpha, attack_mode="gravity"),
    args.hallucinate,
    args.strong_antagonist,
)
# %% Generate Agent.
agent = get_agent(
    args.agent,
    environment,
    protagonist_dynamical_model=protagonist_dynamical_model,
    antagonist_dynamical_model=antagonist_dynamical_model,
    reward_model=PendulumReward(),
    horizon=40,
)
agent.policy.solver.num_samples = 500

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
