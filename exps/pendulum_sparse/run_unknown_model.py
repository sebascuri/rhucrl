"""Python Script Template."""

from rllib.model import TransformedModel
from rllib.util.utilities import set_random_seed

from exps.pendulum.utilities import PendulumReward, pendulum_reset, pendulum_reward
from exps.utilities import get_command_line_parser
from rhucrl.agent import AdversarialMPCAgent, RARLAgent, ZeroSumAgent
from rhucrl.environment.wrappers import (
    HallucinationWrapper,
    ResetWrapper,
    RewardWrapper,
)
from rhucrl.model import HallucinatedModel
from rhucrl.utilities.training import train_adversarial_agent
from rhucrl.utilities.util import get_agent, get_default_models, get_environment

parser = get_command_line_parser()
parser.set_defaults(
    environment="PendulumAdvEnv-v0",
    agent="AdversarialMPC",
    attack_mode="gravity",
    # protagonist_name="STEVE",
    # antagonist_name="STEVE",
    alpha=0.1,
    clip_gradient_val=10.0,
    hallucinate=True,
    strong_antagonist=True,
    exploration_episodes=10,
    num_steps=1,
)
args = parser.parse_args()
args.agent = "AdversarialMPC"

arg_dict = vars(args)

set_random_seed(args.seed)
# %% Generate environment.
environment = get_environment(
    environment=arg_dict.pop("environment"),
    adversarial_wrapper=arg_dict.pop("adversarial_wrapper"),
    alpha=args.alpha,
    seed=args.seed,
    attack_mode=args.attack_mode,
)
environment.add_wrapper(RewardWrapper, reward_function=pendulum_reward)
environment.add_wrapper(ResetWrapper, reset_function=pendulum_reset)

# %% Generate Models.
protagonist_dynamical_model, antagonist_dynamical_model = get_default_models(
    environment, args.hallucinate, args.strong_antagonist
)
print(protagonist_dynamical_model.dim_state, protagonist_dynamical_model.dim_action)
print(antagonist_dynamical_model.dim_state, antagonist_dynamical_model.dim_action)

reward_model = PendulumReward()

# %% Generate Agent.
agent = get_agent(
    arg_dict.pop("agent"),
    environment=environment,
    protagonist_dynamical_model=protagonist_dynamical_model,
    antagonist_dynamical_model=antagonist_dynamical_model,
    reward_model=reward_model,
    **arg_dict,
)
for agent_ in agent.agents:
    print(agent_.comment)
    print(agent_.policy.dim_state, agent_.policy.dim_action)
    if hasattr(agent_, "dynamical_model"):
        print(agent_.dynamical_model.dim_state, agent_.dynamical_model.dim_action)
