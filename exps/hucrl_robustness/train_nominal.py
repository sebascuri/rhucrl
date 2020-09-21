"""Python Script Template."""
from exps.run import evaluate, init_experiment, train_all
from exps.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(
    environment="MBHalfCheetah-v0",
    agent="RARL",
    alpha=0,
    train_episodes=500,
    train_antagonist_episodes=500,
    eval_episodes=10,
)
args = parser.parse_args()

agent, environment = init_experiment(args)
train_all(agent, environment, args)
evaluate(agent, environment, args)