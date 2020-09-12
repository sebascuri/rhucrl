"""Python Script Template."""
from exps.humanoid.utilities import HumanoidReward, HumanoidTermination
from exps.run import run
from exps.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(environment="Humanoid-v4")
args = parser.parse_args()
run(args, reward_model=HumanoidReward(), termination_model=HumanoidTermination())
