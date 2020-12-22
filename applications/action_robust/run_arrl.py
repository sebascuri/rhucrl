"""Python Script Template."""
import argparse
import json

import torch
from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from hucrl.model.hallucinated_model import HallucinatedModel
from rllib.model.transformed_model import TransformedModel
from rllib.util.training.agent_training import train_agent
from rllib.util.utilities import set_random_seed

from applications.util import get_agent, parse_config_file
from rhucrl.agent import ADVERSARIAL_AGENTS, AR_AGENTS
from rhucrl.agent.antagonist_agent import AntagonistAgent
from rhucrl.environment import AdversarialEnv
from rhucrl.environment.wrappers import (
    NoisyActionRobustWrapper,
    ProbabilisticActionRobustWrapper,
)

parser = argparse.ArgumentParser("Action Robust RL")

parser.add_argument("--alpha", type=float, default=0.1, help="Adversarial power.")
parser.add_argument("--agent", type=str, default="RHUCRL", help="Agent name.")
parser.add_argument(
    "--kind",
    type=str,
    default="probabilistic",
    choices=["probabilistic", "noisy"],
    help="Action Robust class.",
)
parser.add_argument("--seed", type=int, default=0, help="random seed.")
parser.add_argument("--num-threads", type=int, default=1, help="Number of threads.")

parser.add_argument(
    "--env-config",
    type=str,
    default="config/envs/half_cheetah.yaml",
    help="Environment config file.",
)
parser.add_argument(
    "--agent-config",
    type=str,
    default="config/agents/sac.yaml",
    help="Agent config file.",
)

args = parser.parse_args()
env_args = parse_config_file(args.env_config)
agent_args = parse_config_file(args.agent_config)
agent_args.update(**{"kind": args.kind, "alpha": args.alpha})
agent_config = args.agent_config.split(".")[-2].split("/")[-1]
name = f"{env_args['name']}_{args.alpha}_{args.agent}_{agent_config}_{args.seed}"
set_random_seed(seed=args.seed)
torch.set_num_threads(args.num_threads)

# Define environment
if args.kind == "probabilistic":
    wrapper = ProbabilisticActionRobustWrapper
elif args.kind == "noisy":
    wrapper = NoisyActionRobustWrapper
else:
    raise argparse.ArgumentError(f"{args.kind} wrongly parsed.")

environment = AdversarialEnv(env_args["name"], seed=args.seed)
if args.agent in ADVERSARIAL_AGENTS:
    environment.add_wrapper(wrapper, alpha=args.alpha)

if args.agent in ["RHUCRL", "BestResponse"] or agent_args.get("beta", 0.0) > 0.0:
    dynamical_model = HallucinatedModel.default(
        environment, beta=agent_args.get("beta", 1.0)
    )
    if args.agent != "ActionRobustHUCRL":
        environment.add_wrapper(HallucinationWrapper)
else:
    dynamical_model = TransformedModel.default(environment)

# Define agent
agent = get_agent(
    environment,
    agent_name=args.agent,
    dynamical_model=dynamical_model,
    **agent_args,
)
if args.agent in AR_AGENTS:
    environment.add_wrapper(wrapper, alpha=args.alpha)
    if args.agent == "ActionRobustHUCRL":
        environment.add_wrapper(HallucinationWrapper)

train_agent(
    agent=agent,
    environment=environment,
    num_episodes=agent_args["num_episodes"],
    max_steps=env_args["max_steps"],
    print_frequency=0,
)

with open(f"{name}.json", "w") as f:
    json.dump(agent.logger.statistics, f)

if args.agent not in ADVERSARIAL_AGENTS and args.agent not in AR_AGENTS:
    environment.add_wrapper(wrapper, alpha=args.alpha)
robust_antagonist = AntagonistAgent.default(
    environment=environment, protagonist_agent=agent, base_agent_name="SAC"
)
train_agent(
    robust_antagonist,
    environment=environment,
    num_episodes=agent_args["num_episodes"],
    max_steps=env_args["max_steps"],
    print_frequency=0,
)
with open(f"{name}_robust.json", "w") as f:
    json.dump(robust_antagonist.logger.statistics, f)
