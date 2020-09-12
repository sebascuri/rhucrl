"""Python Script Template."""
from exps.run import run
from exps.utilities import get_command_line_parser
from exps.walker2d.utilities import Walker2dReward, Walker2dTermination

parser = get_command_line_parser()
parser.set_defaults(environment="Walker2d-v4")
args = parser.parse_args()
run(args, reward_model=Walker2dReward(), termination_model=Walker2dTermination())
