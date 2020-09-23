"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

runner = init_runner("EvaluateKnownModel", wall_time=24 * 60, num_threads=4)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "evaluate_nominal.py"

ENVIRONMENTS = [
    # "MBHalfCheetah-v0",
    "PendulumSwingUp-v0",
    "MBHopper-v0",
    "MBWalker2d-v0",
]

commands = make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0, "hallucinate": False},
    common_hyper_args={
        "seed": [0],
        "environment": ENVIRONMENTS,
        "protagonist-name": ["SAC", "MPO", "PPO"],
    },
)

commands += make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0, "hallucinate": [True, False]},
    common_hyper_args={
        "seed": [0],
        "environment": ENVIRONMENTS,
        "protagonist-name": ["BPTT"],
        "num-steps": [1, 4, 8],
    },
)

commands += make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0, "hallucinate": [True, False], "protagonist-name": "MVE"},
    common_hyper_args={
        "seed": [0],
        "environment": ENVIRONMENTS,
        "base-agent-name": ["SAC", "MPO", "BPTT", "PPO"],
        "num-steps": [1, 4, 8],
    },
)

runner.run(commands)
