"""Python Script Template."""
import os

from rhucrl_experiments.run import evaluate, init_experiment, train_antagonist
from rhucrl_experiments.utilities import get_command_line_parser


def get_nominal_path(protagonist_agent):
    """Get path of nominal agent."""
    pkl_dir = None

    log_dir = protagonist_agent.logger.log_dir.split("/")
    base_dir = "/".join(log_dir[:2])
    split_log_dir = log_dir[2].split(" ")
    protagonist_name = log_dir[3]

    nominal_agent_dir = f"{split_log_dir[0]} NoisyAction 0.0_"
    agents_dir = [*filter(lambda x: nominal_agent_dir in x, os.listdir(base_dir))]

    for agent_dir in agents_dir:
        if protagonist_name not in os.listdir(base_dir + "/" + agent_dir):
            continue
        else:
            base = "/".join([base_dir, agent_dir, protagonist_name])
            pkl_dir = "/".join([base, os.listdir(base)[0]])

    if pkl_dir is None:
        raise RuntimeError("Nominal agent not found!")
    pkl = [
        *filter(
            lambda x: "_best.pkl" in x and "Protagonist" not in x, os.listdir(pkl_dir)
        )
    ][0]

    protagonist_path = "/".join([pkl_dir, pkl])
    return protagonist_path


parser = get_command_line_parser()
parser.set_defaults(
    environment="MBReacher2d-v0",
    agent="RARL",
    alpha=0.1,
    train_episodes=50,
    train_antagonist_episodes=50,
    eval_episodes=10,
)
args = parser.parse_args()
agent, environment = init_experiment(args)
protagonist = agent.agents["Protagonist"]
agent.load_protagonist(get_nominal_path(protagonist))
agent.exploration_episodes = 0
agent.exploration_steps = 0

protagonist.eval()
train_antagonist(agent=agent, environment=environment, args=args)
evaluate(agent=agent, environment=environment, args=args)
