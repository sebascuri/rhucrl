"""Experiment utility files."""
import argparse
from importlib import import_module

import yaml
from hucrl.environment.hallucination_wrapper import HallucinationWrapper

from rhucrl.environment import AdversarialEnv
from rhucrl.environment.wrappers import MujocoAdversarialWrapper


def parse_config_file(file_dir):
    """Parse configuration file."""
    with open(file_dir, "r") as file:
        args = yaml.safe_load(file)
    return args


def get_environment(name, alpha, force_body_names=("torso",)):
    """Get environment."""
    environment = AdversarialEnv(name)
    environment.add_wrapper(
        MujocoAdversarialWrapper, alpha=alpha, force_body_names=force_body_names
    )
    environment.add_wrapper(HallucinationWrapper)

    return environment


def get_agent(environment, agent_name, *args, **kwargs):
    """Get environment."""
    try:
        agent_module = import_module("rhucrl.agent")
        agent = getattr(agent_module, f"{agent_name}Agent").default(
            environment, *args, **kwargs
        )
    except AttributeError:
        agent_module = import_module("rllib.agent")
        agent = getattr(agent_module, f"{agent_name}Agent").default(
            environment, *args, **kwargs
        )
    return agent


def get_parser():
    """Get parser."""
    parser = argparse.ArgumentParser(description="Parameters for H-UCRL.")
    parser.add_argument(
        "--agent",
        type=str,
        default="RHUCRL",
        choices=["RARL", "RAP", "MaxiMin", "BestResponse", "RHUCRL"],
    )
    parser.add_argument("--base-agent", type=str, default="BPTT")
    parser.add_argument("--config-file", type=str, default="config/bptt.yaml")

    parser.add_argument("--environment", type=str, default="MBHalfCheetah-v0")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--action-cost", type=float, default=0.1)

    parser.add_argument("--hallucinate", type=bool, action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=2)

    parser.add_argument("--max-steps", type=int, default=1000)

    parser.add_argument("--train-episodes", type=int, default=200)

    parser.add_argument("--antagonist-episodes", type=int, default=200)

    parser.add_argument("--render", action="store_true", default=False)

    return parser
