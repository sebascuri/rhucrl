"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

runner = init_runner("KnownModel.", wall_time=24 * 60, num_threads=4)
cwd = os.getcwd()
script = "train_nominal.py"

environments = ["MBHalfCheetah-v0"]

commands = make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0, "hallucinate": False, "train-episodes": 20},
    common_hyper_args={"seed": [0], "protagonist-name": ["SAC"]},
)

runner.run(commands)
