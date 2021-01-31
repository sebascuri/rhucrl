"""Launch Inverted Pendulum experiments."""

import os
from itertools import product

from lsf_runner import init_runner, make_commands

cwd = os.path.dirname(os.path.realpath(__file__))
script = "run.py"

AGENTS = ["RHUCRL", "HUCRL", "baseline"]
ROBUSTNESS = ["parameter"]
ALPHAS = [0]

runner = init_runner(f"Inverted Pendulum", num_threads=1, memory=4096)

commands = []
for agent, robustness in product(AGENTS, ROBUSTNESS):
    commands += make_commands(
        script, base_args={"agent": agent, "robustness": robustness}
    )
runner.run(commands)
