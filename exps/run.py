"""Python Script Template."""

from rllib.util.utilities import set_random_seed

from exps.utilities import get_command_line_parser
from rhucrl.environment import AdversarialEnv
from rhucrl.utilities.training import train_adversarial_agent
from rhucrl.utilities.util import (
    get_agent,
    get_default_models,
    wrap_adversarial_environment,
)


def run(args, dynamical_model=None, reward_model=None, termination_model=None):
    """Run main function with arguments."""
    if args.antagonist_name is None:
        args.antagonist_name = args.protagonist_name

    arg_dict = vars(args)
    set_random_seed(args.seed)

    # %% Get environment.
    environment = AdversarialEnv(env_name=arg_dict.pop("environment"), seed=args.seed)
    wrap_adversarial_environment(
        environment, args.adversarial_wrapper, args.alpha, args.force_body_names
    )

    # %% Generate Models.
    protagonist_dynamical_model, antagonist_dynamical_model = get_default_models(
        environment,
        known_model=dynamical_model,
        hallucinate=args.hallucinate,
        strong_antagonist=args.strong_antagonist,
    )

    # %% Initialize agent.
    agent = get_agent(
        arg_dict.pop("agent"),
        environment=environment,
        protagonist_dynamical_model=protagonist_dynamical_model,
        antagonist_dynamical_model=antagonist_dynamical_model,
        reward_model=reward_model,
        termination_model=termination_model,
        **arg_dict,
    )

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
    return agent


if __name__ == "__main__":
    parser = get_command_line_parser()
    run(args=parser.parse_args())
