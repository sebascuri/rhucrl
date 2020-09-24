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
    agent="ZeroSum", train_episodes=200, train_antagonist_episodes=200, eval_episodes=10
)
args = parser.parse_args()

agent, environment = init_experiment(args)
train_all(agent, environment, args)
train_antagonist(agent, environment, args)
evaluate(agent, environment, args)
