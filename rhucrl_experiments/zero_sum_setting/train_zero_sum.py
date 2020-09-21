"""Python Script Template."""
from rhucrl_experiments.run import (
    evaluate,
    init_experiment,
    train_all,
    train_antagonist,
)
from rhucrl_experiments.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(
    environment="MBHalfCheetah-v0",
    agent="ZeroSum",
    protagonist_name="MVE",
    adversarial_wrapper="noisy_action",
    alpha=0.1,
    exploration_episodes=0,
    train_episodes=50,
    train_antagonist_episodes=5,
    eval_episodes=10,
    hallucinate=True,
    strong_antagonist=True,
)
args = parser.parse_args()

agent, environment = init_experiment(args)
train_all(agent, environment, args)
train_antagonist(agent, environment, args)
evaluate(agent, environment, args)
