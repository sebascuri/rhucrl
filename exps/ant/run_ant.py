"""Python Script Template."""
from exps.ant.utilities import AntReward, AntTermination
from exps.run import run
from exps.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(environment="Ant-v4")
args = parser.parse_args()
run(args, reward_model=AntReward(), termination_model=AntTermination())
