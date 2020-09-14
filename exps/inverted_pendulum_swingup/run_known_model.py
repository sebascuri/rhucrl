"""Python Script Template."""
from exps.inverted_pendulum_swingup.utilities import PendulumModel, PendulumReward
from exps.run import run
from exps.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(
    environment="Pendulum-v1",
    agent="AdversarialMPC",
    adversarial_wrapper="adversarial_pendulum",
    alpha=0.1,
    force_body_names=["gravity", "mass"],
    horizon=40,
    train_episodes=1,
    exploration_episodes=0,
    max_steps=200,
    render=True,
)
args = parser.parse_args()
model = PendulumModel(alpha=args.alpha, force_body_names=args.force_body_names)
reward_model = PendulumReward()

agent = run(args, dynamical_model=model, reward_model=reward_model)
