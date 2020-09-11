"""Python Script Template."""

from rllib.util.utilities import set_random_seed

from exps.half_cheetah.utilities import HalfCheetahReward
from exps.utilities import get_command_line_parser
from rhucrl.environment.wrappers import MujocoAdversarialWrapper
from rhucrl.utilities.training import train_adversarial_agent
from rhucrl.utilities.util import get_agent, get_default_models, get_environment

parser = get_command_line_parser()
parser.set_defaults(
    environment="HalfCheetah-v4",
    agent="RARL",
    protagonist_name="STEVE",
    antagonist_name="STEVE",
    alpha=0.1,
    clip_gradient_val=10.0,
    hallucinate=True,
    strong_antagonist=True,
    exploration_episodes=10,
    num_steps=1,
)
args = parser.parse_args()
arg_dict = vars(args)

set_random_seed(args.seed)
# %% Generate environment.
environment = get_environment(environment=arg_dict.pop("environment"), seed=args.seed)
environment.add_wrapper(
    MujocoAdversarialWrapper,
    force_body_names=["torso", "bfoot", "ffoot"],
    alpha=args.alpha,
)

# %% Generate Models.
protagonist_dynamical_model, antagonist_dynamical_model = get_default_models(
    environment, hallucinate=args.hallucinate, strong_antagonist=args.strong_antagonist
)
reward_model = HalfCheetahReward()

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
