"""Python Script Template."""

from rllib.util.utilities import set_random_seed

from exps.utilities import get_agent, get_command_line_parser, get_environment
from rhucrl.utilities.training import train_adversarial_agent


def run(args):
    """Run main function with arguments."""
    # %% Set Random seeds.
    set_random_seed(args.seed)

    # %% Get environment.
    environment = get_environment(args)

    # %% Initialize agent.
    agent = get_agent(args, environment)

    # %% Train Agent.
    train_adversarial_agent(
        mode="both",
        agent=agent,
        environment=environment,
        num_episodes=args.train_episodes,
        max_steps=args.max_steps,
        print_frequency=1,
        eval_frequency=args.eval_frequency,
        render=args.render,
    )

    # %% Train Antagonist only.
    train_adversarial_agent(
        mode="antagonist",
        agent=agent,
        environment=environment,
        num_episodes=args.train_episodes,
        max_steps=args.max_steps,
        print_frequency=1,
        eval_frequency=args.eval_frequency,
        render=args.render,
    )


if __name__ == "__main__":
    parser = get_command_line_parser()
    run(args=parser.parse_args())
