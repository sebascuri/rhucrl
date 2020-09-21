"""Python Script Template."""

from rllib.util.training.agent_training import evaluate_agent
from rllib.util.utilities import set_random_seed

from rhucrl.environment import AdversarialEnv
from rhucrl.environment.wrappers import HallucinationWrapper
from rhucrl.utilities.training import train_adversarial_agent
from rhucrl.utilities.util import get_agent, wrap_adversarial_environment
from rhucrl_experiments.utilities import get_command_line_parser


def init_experiment(args, env_kwargs=None, **kwargs):
    """Initialize experiment to get agent and environment."""
    env_kwargs = dict() if env_kwargs is None else env_kwargs
    if args.antagonist_name is None:
        args.antagonist_name = args.protagonist_name

    arg_dict = vars(args)
    arg_dict.update(kwargs)
    set_random_seed(args.seed)

    # %% Get environment.
    environment = AdversarialEnv(
        env_name=arg_dict.pop("environment"), seed=args.seed, **env_kwargs
    )
    wrap_adversarial_environment(
        environment, args.adversarial_wrapper, args.alpha, args.force_body_names
    )

    # %% Initialize agent.
    agent = get_agent(arg_dict.pop("agent"), environment=environment, **arg_dict)

    # %% Add Hallucination wrapper.
    if args.hallucinate:
        environment.add_wrapper(HallucinationWrapper)

    return agent, environment


def run(args, env_kwargs=None, **kwargs):
    """Run main function with arguments."""
    # %% Initialize experiment.
    agent, environment = init_experiment(args, env_kwargs, **kwargs)

    # %% Train Agent.
    train_all(agent, environment, args)

    # %% Train Antagonist only.
    train_antagonist(agent, environment, args)

    # %% Evaluate agents
    evaluate(agent, environment, args)


def train_all(agent, environment, args):
    """Train all agents."""
    train_adversarial_agent(
        mode="both",
        agent=agent,
        environment=environment,
        num_episodes=args.train_episodes,
        max_steps=args.max_steps,
        print_frequency=1,
        eval_frequency=args.eval_frequency,
        render=args.render_train,
    )


def train_antagonist(agent, environment, args):
    """Train antagonist agents."""
    train_adversarial_agent(
        mode="antagonist",
        agent=agent,
        environment=environment,
        num_episodes=args.train_antagonist_episodes,
        max_steps=args.max_steps,
        print_frequency=1,
        eval_frequency=args.eval_frequency,
        render=args.render_train,
    )


def evaluate(agent, environment, args):
    """Evaluate agents."""
    evaluate_agent(
        agent=agent,
        environment=environment,
        num_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        render=args.render_eval,
    )


if __name__ == "__main__":
    parser = get_command_line_parser()
    run(args=parser.parse_args())
