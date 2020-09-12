"""Python Script Template."""
from exps.inverted_pendulum.utilities import PendulumReward, PendulumTermination
from exps.run import run
from exps.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(
    environment="InvertedPendulum-v2",
    force_body_names=["cart", "pole"],
    clip_gradient_val=10,
)
args = parser.parse_args()
run(args, reward_model=PendulumReward(), termination_model=PendulumTermination())
