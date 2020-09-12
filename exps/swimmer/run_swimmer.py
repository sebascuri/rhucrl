"""Python Script Template."""
from exps.run import run
from exps.swimmer.utilities import SwimmerReward, SwimmerTermination
from exps.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(environment="Swimmer-v4")
args = parser.parse_args()
run(args, reward_model=SwimmerReward(), termination_model=SwimmerTermination())
