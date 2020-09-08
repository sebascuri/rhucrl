"""Python Script Template."""
import argparse
import importlib
from typing import Any

from gym.envs import registry
from rllib.agent import AGENTS as BASE_AGENTS

from rhucrl.agent import AGENTS as ROBUST_AGENTS
from rhucrl.agent.adversarial_agent import AdversarialAgent
from rhucrl.environment import AdversarialEnv
from rhucrl.environment.wrappers import (
    NoisyActionRobustWrapper,
    ProbabilisticActionRobustWrapper,
)


def get_command_line_parser():
    """Get command line parser."""
    parser = argparse.ArgumentParser("Run experiment on RLLib.")
    parser.add_argument(
        "--environment",
        default="HalfCheetahAdvEnv-v0",
        type=str,
        help="Environment name.",
        choices=list(registry.env_specs.keys()),
    )

    parser.add_argument(
        "--agent",
        default="RARL",
        type=str,
        help="Robust agent name.",
        choices=ROBUST_AGENTS,
    )
    parser.add_argument(
        "--protagonist",
        default="SAC",
        type=str,
        help="Protagonist agent name.",
        choices=BASE_AGENTS,
    )
    parser.add_argument(
        "--antagonist",
        default=None,
        type=str,
        help="Antagonist agent name.",
        choices=BASE_AGENTS,
    )
    parser.add_argument("--alpha", default=0.1, type=float, help="Antagonist power.")
    parser.add_argument(
        "--adversarial-wrapper",
        default=None,
        type=str,
        help="Wrapper.",
        choices=["noisy_action", "probabilistic_action", None],
    )

    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps.")
    parser.add_argument(
        "--train-episodes", type=int, default=200, help="Total number of episodes."
    )
    parser.add_argument(
        "--exploration-steps", type=int, default=0, help="Exploit after n steps."
    )
    parser.add_argument(
        "--exploration-episodes", type=int, default=0, help="Exploit after n episodes."
    )

    parser.add_argument(
        "--eval-frequency", type=int, default=20, help="Evaluate every n episodes."
    )
    parser.add_argument("--render", action="store_true", default=False)

    return parser


def get_environment(arguments: argparse.Namespace, **kwargs: Any) -> AdversarialEnv:
    """Get environment."""
    if arguments.adversarial_wrapper == "noisy_action":
        environment = AdversarialEnv(arguments.environment, seed=arguments.seed)
        environment.add_wrapper(NoisyActionRobustWrapper, alpha=arguments.alpha)
    elif arguments.adversarial_wrapper == "probabilistic_action":
        environment = AdversarialEnv(arguments.environment, seed=arguments.seed)
        environment.add_wrapper(ProbabilisticActionRobustWrapper, alpha=arguments.alpha)
    else:
        environment = AdversarialEnv(
            arguments.environment, seed=arguments.seed, alpha=arguments.alpha, **kwargs
        )
    return environment


def get_agent(
    arguments: argparse.Namespace, environment: AdversarialEnv, **kwargs: Any
) -> AdversarialAgent:
    """Get default agent."""
    agent_module = importlib.import_module("rhucrl.agent")
    agent = getattr(agent_module, f"{arguments.agent}Agent").default(
        environment,
        exploration_steps=arguments.exploration_steps,
        exploration_episodes=arguments.exploration_episodes,
        **kwargs,
    )
    return agent
