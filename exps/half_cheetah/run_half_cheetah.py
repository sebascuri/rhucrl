"""Python Script Template."""
from exps.half_cheetah.utilities import HalfCheetahReward, HalfCheetahTermination
from exps.run import run
from exps.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(environment="HalfCheetah-v4")
args = parser.parse_args()
run(args, reward_model=HalfCheetahReward(), termination_model=HalfCheetahTermination())
