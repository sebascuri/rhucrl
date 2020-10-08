"""Python Script Template."""
import argparse
import os

from rllib.util.utilities import set_random_seed

from rhucrl.environment import AdversarialEnv
from rhucrl.environment.wrappers import HallucinationWrapper
from rhucrl.utilities.util import get_agent, wrap_adversarial_environment

ENVIRONMENTS = [
    "MBHalfCheetah-v0",
    "MBHopper-v0",
    "MBWalker2d-v0",
    "MBSwimmer-v0",
    "MBReacher3d-v0",
]


def get_command_line_parser():
    """Get command line parser."""
    parser = argparse.ArgumentParser("Evaluate RH-UCRL.")
    parser.add_argument("--algorithm", default="RHUCRL", type=str)
    parser.add_argument(
        "--environment", default="MBHalfCheetah-v0", type=str, choices=ENVIRONMENTS
    )
    parser.add_argument("--base-dir", default="../../runs/RARLAgent/", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-runs", default=10, type=int)
    return parser


def _parse_path(path):
    environment, *wrapper, code = path.split(" ")
    environment = environment.split("/")[-1]
    code_split = code.split("_")
    alpha = code_split[0]
    time = "_".join(code_split[1:])
    return environment, wrapper, alpha, time


def _get_protagonist_path(agent_path):
    protagonist_ = list(filter(lambda x: "Protagonist" in x, os.listdir(agent_path)))[0]
    protagonist_path = os.path.join(agent_path, protagonist_)
    protagonist_path = os.path.join(protagonist_path, os.listdir(protagonist_path)[0])
    p_ = list(filter(lambda x: "Protagonist" in x, os.listdir(protagonist_path)))[0]
    return os.path.join(protagonist_path, p_)


def get_trained_protagonist_path(agent, results_dir=None):
    """Get trained agent path."""
    path = agent.logger.log_dir
    if results_dir is not None:
        path = results_dir + path
    base_path = "/".join(path.split("/")[:-1])

    environment, wrapper, alpha, time = _parse_path(path)
    protagonist_name = list(filter(lambda x: "Protagonist" in x, os.listdir(path)))[0]

    for path_ in os.listdir(base_path):
        environment_, wrapper_, alpha_, time_ = _parse_path(path_)
        path_ = base_path + "/" + path_
        name_ = list(filter(lambda x: "Protagonist" in x, os.listdir(path)))[0]

        if (
            environment == environment_
            and wrapper_ == wrapper
            and alpha == alpha_
            and name_ == protagonist_name
            and time != time_
        ):
            print("a match is found :).")
            return _get_protagonist_path(path_)
        else:
            continue


def init_experiment(args, agent_dir, **kwargs):
    """Initialize experiment."""
    if args.antagonist_name is None:
        args.antagonist_name = args.protagonist_name

    arg_dict = vars(args)
    arg_dict.update(kwargs)

    set_random_seed(args.seed)
    environment = AdversarialEnv(env_name=arg_dict.pop("environment"), seed=args.seed)
    wrap_adversarial_environment(
        environment, args.adversarial_wrapper, args.alpha, args.force_body_names
    )
    agent = get_agent(arg_dict.pop("agent"), environment=environment, **arg_dict)

    environment.pop_wrapper()

    if args.hallucinate:
        environment.add_wrapper(HallucinationWrapper)

    agent.load_protagonist(_get_protagonist_path(agent_dir))
    agent.exploration_episodes = 0
    agent.exploration_steps = 0

    protagonist = agent.agents["Protagonist"]
    protagonist.eval()
    protagonist.policy.only_protagonist = True
    return protagonist, environment
