"""Python Script Template."""
from exps.run import run
from exps.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(
    environment="PendulumSwingUp-v0",
    agent="RARL",
    adversarial_wrapper="adversarial_pendulum",
    force_body_names=["gravity", "mass"],
    train_episodes=0,
    train_antagonist_episodes=0,
    eval_episodes=10,
)
args = parser.parse_args()
agent = run(args)
