"""Launch Inverted Pendulum experiments."""

import os
from itertools import product

from lsf_runner import init_runner, make_commands

cwd = os.path.dirname(os.path.realpath(__file__))
script = "run.py"

AGENTS = ["RHUCRL", "HUCRL"]  # , "baseline"]
ROBUSTNESS = ["adversarial", "action", "parameter"]
ALPHAS = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


for agent, robustness in product(AGENTS, ROBUSTNESS):
    runner = init_runner(
        f"AdversarialRL_{agent}", wall_time=24 * 60, num_threads=1, memory=4096
    )
    commands = make_commands(
        script,
        base_args={"agent": agent, "robustness": robustness},
        common_hyper_args={"alpha": ALPHAS},
    )
    # print(commands)
    runner.run_batch(commands)
