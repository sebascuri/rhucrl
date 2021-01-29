"""RH-UCRL in inverted pendulum."""
import argparse
import json

import numpy as np
from rllib.model.transformed_model import TransformedModel
from rllib.util.training.agent_training import evaluate_agent, train_agent
from rllib.util.utilities import set_random_seed

from applications.inverted_pendulum.utilities import PendulumModel
from rhucrl.agent import ActionRobustAgent, AdversarialMPCAgent, EPOPTAgent, RARLAgent
from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.environment.wrappers import (
    AdversarialPendulumWrapper,
    NoisyActionRobustWrapper,
)


def init_experiment(args, env_kwargs=None, **kwargs):
    """Initialize experiment to get agent and environment."""
    if args.robustness == "parameter":
        args.alpha = 0

    env_kwargs = dict() if env_kwargs is None else env_kwargs

    arg_dict = vars(args).copy()
    arg_dict.update(kwargs)
    set_random_seed(args.seed)

    # Get environment.
    environment = AdversarialEnv(
        env_name="PendulumSwingUp-v0", seed=args.seed, **env_kwargs
    )
    reward_model = environment.env.reward_model()

    dynamical_model = TransformedModel(
        PendulumModel(
            alpha=args.alpha, force_body_names=("mass",), wrapper=args.robustness
        ),
        transformations=[],
    )

    if args.robustness == "adversarial":
        environment.add_wrapper(
            AdversarialPendulumWrapper, alpha=args.alpha, force_body_names=("mass",)
        )
    elif args.robustness == "action":
        environment.add_wrapper(NoisyActionRobustWrapper, alpha=args.alpha)
    elif args.robustness == "parameter" and args.agent == "RHUCRL":
        dynamical_model.base_model.mass = args.alpha * 30 / 7 + 2 / 7

    # Initialize agent.
    if args.agent == "HUCRL" or args.agent == "RHUCRL":
        agent = AdversarialMPCAgent.default(
            environment=environment,
            nominal_model=args.agent == "HUCRL",
            reward_model=reward_model,
            dynamical_model=dynamical_model,
            horizon=40,
            exploration_steps=0,
            exploration_episodes=0,
        )
    else:
        if args.wrapper == "adversary":
            agent = RARLAgent.default(
                environment, exploration_steps=0, exploration_episodes=0
            )
        elif args.wrapper == "action":
            agent = ActionRobustAgent.default(
                environment,
                kind="probabilistic",
                exploration_steps=0,
                exploration_episodes=0,
            )
        else:
            agent = EPOPTAgent.default(
                environment, exploration_steps=0, exploration_episodes=0
            )

        train_agent(
            agent=agent,
            environment=environment,
            num_episodes=200,
            max_steps=500,
            print_frequency=0,
        )

    agent.logger.save_hparams(vars(args))
    return agent, environment


def run(args, env_kwargs=None, **kwargs):
    """Run main function with arguments."""
    # Initialize experiment.
    agent, environment = init_experiment(args, env_kwargs, **kwargs)

    name = f"{args.agent}_{args.robustness}_{args.alpha}"

    agent.eval()
    if args.robustness == "parameter":
        # Sweep the mass.
        for m in np.linspace(0.5, 2.0, 11):
            for _ in range(5):
                agent.logger.update(**{f"mass": m})
                environment.env.m = m
                evaluate_agent(
                    agent=agent,
                    environment=environment,
                    num_episodes=1,
                    max_steps=500,
                    render=False,
                )

    else:
        evaluate_agent(
            agent=agent,
            environment=environment,
            num_episodes=5,
            max_steps=500,
            render=False,
        )

    with open(f"{name}.json", "w") as f:
        json.dump(agent.logger.statistics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Inverted Pendulum with RH-UCRL.")
    parser.add_argument(
        "--agent", default="RHUCRL", type=str, choices=["RHUCRL", "HUCRL", "baseline"]
    )
    parser.add_argument("--alpha", default=0.1, type=float, help="Antagonist power.")
    parser.add_argument(
        "--robustness",
        default="adversarial",
        type=str,
        help="Wrapper.",
        choices=["adversarial", "action", "parameter"],
    )

    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    run(parser.parse_args())
