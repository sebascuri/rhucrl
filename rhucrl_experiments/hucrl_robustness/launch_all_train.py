"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

from rhucrl_experiments.get_environment import get_environment

runner = init_runner("TrainKnownModel", wall_time=24 * 60, num_threads=4)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_nominal.py"

ENVIRONMENTS = get_environment()
print(ENVIRONMENTS)
SEEDS = [0]
TRAIN_EPISODES = 200
TRAIN_ANTAGONIST_EPISODES = 0
EVALUATE_EPISODES = 10
BASE_ARGS = {
    "alpha": 0,
    "train-episodes": 200,
    "train-antagonist-episodes": 0,
    "eval-episodes": 10,
}

AGENTS = ["TD3", "MPO", "PPO", "VMPO"]
commands = make_commands(
    f"{cwd}/{script}",
    base_args={**BASE_ARGS, "hallucinate": False},
    common_hyper_args={
        "seed": SEEDS,
        "environment": ENVIRONMENTS,
        "protagonist-name": AGENTS,
    },
)

commands += make_commands(
    f"{cwd}/{script}",
    base_args={**BASE_ARGS, "num-steps": 1},
    common_hyper_args={
        "seed": SEEDS,
        "environment": ENVIRONMENTS,
        "hallucinate": [True, False],
        "protagonist-name": ["BPTT"],
    },
)

commands += make_commands(
    f"{cwd}/{script}",
    base_args={**BASE_ARGS, "num-steps": 1},
    common_hyper_args={
        "seed": SEEDS,
        "protagonist-name": ["Dyna", "MVE"],
        "environment": ENVIRONMENTS,
        "hallucinate": [True, False],
        "base-agent": ["TD3", "MPO", "BPTT"],
    },
)

print(len(commands))
runner.run([commands[0]])
