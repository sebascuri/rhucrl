"""Python Script Template."""
import argparse
import json

import torch
from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from hucrl.model.hallucinated_model import HallucinatedModel
from rllib.agent.fixed_policy_agent import FixedPolicyAgent
from rllib.model.transformed_model import TransformedModel
from rllib.policy.constant_policy import ConstantPolicy
from rllib.util.training.agent_training import train_agent
from rllib.util.utilities import set_random_seed

from applications.util import get_agent, parse_config_file
from rhucrl.agent import ADVERSARIAL_AGENTS, DR_AGENTS
from rhucrl.environment import AdversarialEnv
from rhucrl.environment.wrappers import MujocoDomainRandomizationWrapper
from rhucrl.utilities.utilities import evaluate_domain_shift

parser = argparse.ArgumentParser("Robust Domain Randomization RL")

parser.add_argument("--agent", type=str, default="RHUCRL", help="Agent name.")
parser.add_argument("--seed", type=int, default=0, help="random seed.")
parser.add_argument("--num-threads", type=int, default=1, help="Number of threads.")
parser.add_argument(
    "--num-eval-runs", type=int, default=5, help="Number of runs per parameter."
)

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

agent_config = args.agent_config.split(".")[-2].split("/")[-1]
name = f"{env_args['name']}_{args.agent}_{agent_config}_{args.seed}"
set_random_seed(seed=args.seed)
torch.set_num_threads(args.num_threads)

# Define environment
environment = AdversarialEnv(env_args["name"], seed=args.seed)
if args.agent in ADVERSARIAL_AGENTS:
    environment.add_wrapper(
        MujocoDomainRandomizationWrapper,
        mass_names=env_args.get("mass_names", None),
        friction_names=env_args.get("friction_names", None),
    )
if args.agent in ["RHUCRL", "BestResponse"] or agent_args.get("beta", 0.0) > 0:
    dynamical_model = HallucinatedModel.default(
        environment, beta=agent_args.get("beta", 1.0)
    )
    environment.add_wrapper(HallucinationWrapper)
else:
    dynamical_model = TransformedModel.default(environment)

# Define agent
antagonist_policy = ConstantPolicy(
    dim_state=environment.dim_state, dim_action=environment.antagonist_dim_action
)
num_params = len(env_args.get("mass_names", [])) + len(
    env_args.get("friction_names", [])
)

agent = get_agent(
    environment,
    agent_name=args.agent,
    dynamical_model=dynamical_model,
    antagonist_policy=antagonist_policy,
    num_params=num_params,
    **agent_args,
)

if args.agent in DR_AGENTS:
    environment.add_wrapper(
        MujocoDomainRandomizationWrapper,
        mass_names=env_args.get("mass_names", None),
        friction_names=env_args.get("friction_names", None),
    )


train_agent(
    agent=agent,
    environment=environment,
    num_episodes=agent_args["num_episodes"],
    max_steps=env_args["max_steps"],
    print_frequency=0,
)

with open(f"{name}.json", "w") as f:
    json.dump(agent.logger.statistics, f)

try:
    policy = agent.policy.protagonist_policy
except AttributeError:
    policy = agent.policy

eval_agent = FixedPolicyAgent.default(environment, policy=policy)
eval_agent.logger.change_log_dir(name)
evaluate_domain_shift(
    env_args=env_args, agent=eval_agent, seed=args.seed, num_runs=args.num_eval_runs
)
with open(f"{name}_eval.json", "w") as f:
    json.dump(eval_agent.logger.statistics, f)
