"""Python Script Template."""
import argparse
import importlib

from rllib.agent import AGENTS as BASE_AGENTS
from rllib.util.training.agent_training import evaluate_agent
from rllib.util.utilities import set_random_seed

from rhucrl.agent import AGENTS as ROBUST_AGENTS
from rhucrl.environment import ENVIRONMENTS, AdversarialEnv
from rhucrl.utilities.training import train_adversarial_agent


def main(args):
    """Run main function with arguments."""
    # %% Set Random seeds.
    set_random_seed(args.seed)

    # %% Initialize environment.
    environment = AdversarialEnv(args.environment, seed=args.seed)

    # %% Initialize agent.
    agent_module = importlib.import_module("rhucrl.agent")
    agent = getattr(agent_module, f"{args.agent}Agent").default(
        environment,
        exploration_steps=args.exploration_steps,
        exploration_episodes=args.exploration_episodes,
    )

    # %% Train Agent.
    train_adversarial_agent(
        mode="both",
        agent=agent,
        environment=environment,
        num_episodes=args.train_episodes,
        max_steps=args.max_steps,
        print_frequency=1,
        eval_frequency=args.eval_frequency,
    )

    # %% Evaluate Agent.
    evaluate_agent(
        agent=agent,
        environment=environment,
        num_episodes=args.test_episodes,
        max_steps=args.max_steps,
    )

    # %% Train Antagonist only.
    train_adversarial_agent(
        mode="antagonist",
        agent=agent,
        environment=environment,
        num_episodes=args.train_episodes,
        max_steps=args.max_steps,
        print_frequency=1,
        eval_frequency=args.eval_frequency,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run experiment on RLLib.")
    parser.add_argument(
        "--environment",
        default="HalfCheetahAdvEnv-v0",
        type=str,
        help="Environment name.",
        choices=ENVIRONMENTS,
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
    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps.")
    parser.add_argument(
        "--train-episodes", type=int, default=200, help="Training episodes."
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=5, help="Evaluation episodes."
    )
    parser.add_argument(
        "--exploration-steps", type=int, default=0, help="Exploration Steps."
    )
    parser.add_argument(
        "--exploration-episodes", type=int, default=0, help="Exploration Episodes."
    )

    parser.add_argument(
        "--eval-frequency", type=int, default=20, help="Evaluation Frequency."
    )
    parser.add_argument("--render-test", action="store_true", default=False)

    main(parser.parse_args())
