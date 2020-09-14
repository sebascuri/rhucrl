"""Python Script Template."""

from exps.inverted_pendulum_swingup.utilities import PendulumReward
from exps.run import run
from exps.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(
    environment="Pendulum-v1",
    adversarial_wrapper="adversarial_pendulum",
    alpha=0.1,
    force_body_names=["gravity", "mass"],
    max_steps=200,
    exploration_episodes=10,
    model_learning_exploration_episodes=5,
    protagonist_name="STEVE",
    antagonist_name="STEVE",
    clip_gradient_val=10.0,
    hallucinate=True,
    strong_antagonist=True,
)
args = parser.parse_args()
run(args, reward_model=PendulumReward())
