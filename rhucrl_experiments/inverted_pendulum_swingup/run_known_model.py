"""Python Script Template."""
from rhucrl_experiments.inverted_pendulum_swingup.utilities import PendulumModel
from rhucrl_experiments.run import run
from rhucrl_experiments.utilities import get_command_line_parser

parser = get_command_line_parser()
parser.set_defaults(
    environment="PendulumSwingUp-v0",
    agent="AdversarialMPC",
    adversarial_wrapper="probabilistic_action",
    alpha=0.2,
    force_body_names=["gravity", "mass"],
    horizon=40,
    train_episodes=0,
    train_antagonist_episodes=0,
    exploration_episodes=0,
    eval_episodes=5,
    nominal_model=False,
)
args = parser.parse_args()
model = PendulumModel(
    alpha=args.alpha,
    force_body_names=args.force_body_names,
    wrapper=args.adversarial_wrapper,
)
if args.alpha == 0:
    comment = "Nominal"
else:
    if args.nominal_model:
        comment = f"Wrong Model "
    else:
        comment = f"Correct Model "

    comment += f"{args.adversarial_wrapper} {args.alpha}"
agent = run(args, dynamical_model=model, comment=comment)
