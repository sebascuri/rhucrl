"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

runner = init_runner("Known Model.", wall_time=24 * 60, num_threads=4)
cwd = os.getcwd()
script = "evaluate_nominal.py"

environments = [
    "MBHalfCheetah-v0",
    "PendulumSwingUp-v0",
    "MBHopper-v0",
    "MBWalker2d-v0",
]

commands = make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0, "hallucinate": False},
    common_hyper_args={"seed": [0], "protagonist-name": ["SAC", "MPO", "PPO"]},
)

commands += make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0, "hallucinate": [True, False]},
    common_hyper_args={
        "seed": [0],
        "protagonist-name": ["BPTT"],
        "num-steps": [1, 4, 8],
    },
)

commands += make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0, "hallucinate": [True, False], "protagonist-name": "MVE"},
    common_hyper_args={
        "seed": [0],
        "base-agent": ["SAC", "MPO", "BPTT", "PPO"],
        "num-steps": [1, 4, 8],
    },
)

runner.run(commands)
