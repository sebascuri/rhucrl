"""Python Script Template."""
import argparse

from gym.envs import registry
from rllib.agent import AGENTS as BASE_AGENTS
from rllib.agent import MODEL_FREE

from rhucrl.agent import AGENTS as ROBUST_AGENTS


def get_command_line_parser():
    """Get command line parser."""
    parser = argparse.ArgumentParser("Run experiment on RH-UCRL.")
    parser.add_argument(
        "--environment",
        default="HalfCheetah-v2",
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
        "--protagonist-name",
        default="SAC",
        type=str,
        help="Protagonist agent name.",
        choices=BASE_AGENTS,
    )
    parser.add_argument(
        "--antagonist-name",
        default=None,
        type=str,
        help="Antagonist agent name.",
        choices=BASE_AGENTS + [None],
    )
    parser.add_argument(
        "--base-agent",
        default=None,
        type=str,
        help="Base agents for Model-Based Augmented agents.",
        choices=MODEL_FREE,
    )

    parser.add_argument("--nominal-model", action="store_true", default=True)
    parser.add_argument("--hallucinate", action="store_true", default=False)
    parser.add_argument("--strong-antagonist", action="store_true", default=False)
    parser.add_argument("--alpha", default=5.0, type=float, help="Antagonist power.")

    parser.add_argument(
        "--adversarial-wrapper",
        default="external_force",
        type=str,
        help="Wrapper.",
        choices=[
            "noisy_action",
            "probabilistic_action",
            "external_force",
            "adversarial_pendulum",
        ],
    )
    parser.add_argument("--force-body-names", default=["torso"], type=str, nargs="+")

    parser.add_argument(
        "--clip-gradient-val", type=float, default=100.0, help="Maximum gradient norm."
    )
    parser.add_argument(
        "--num-steps", type=int, default=1, help="Number of steps to use the model."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps.")
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=1000,
        help="Total number of episodes to train both protagonist and antagonist.",
    )
    parser.add_argument(
        "--train-antagonist-episodes",
        type=int,
        default=200,
        help="Total number of episodes to train only antagonist after protagonist.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Total number of episodes to train only antagonist.",
    )
    parser.add_argument(
        "--exploration-episodes",
        type=int,
        default=10,
        help="Explore for first n episodes and start policy learning afterwards.",
    )
    parser.add_argument(
        "--exploration-steps",
        type=int,
        default=0,
        help="Explore for first n steps and start policy learning afterwards.",
    )
    parser.add_argument(
        "--model-learn-exploration-episodes",
        type=int,
        default=5,
        help="Explore for first n episodes and start model learning afterwards.",
    )

    parser.add_argument(
        "--eval-frequency", type=int, default=10, help="Evaluate every n episodes."
    )
    parser.add_argument("--render_train", action="store_true", default=False)
    parser.add_argument("--render_eval", action="store_true", default=False)

    return parser
