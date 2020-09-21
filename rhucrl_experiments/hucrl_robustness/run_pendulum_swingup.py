"""Python Script Template."""

from rhucrl_experiments.run import run
from rhucrl_experiments.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(
    agent="RARL",
    environment="PendulumSwingUp-v0",
    adversarial_wrapper="adversarial_pendulum",
    alpha=0,
    force_body_names=["gravity", "mass"],
    max_steps=500,
    exploration_episodes=0,
    model_learn_exploration_episodes=0,
    protagonist_name="BPTT",
    antagonist_name="Random",
    hallucinate=True,
    strong_antagonist=False,
    render=True,
    num_steps=4,
)
args = parser.parse_args()
run(args)
