"""Python Script Template."""
from exps.hopper.utilities import HopperReward, HopperTermination
from exps.run import run
from exps.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(environment="Hopper-v4")
args = parser.parse_args()
run(args, reward_model=HopperReward(), termination_model=HopperTermination())
