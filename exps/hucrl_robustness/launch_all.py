"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

runner = init_runner("Known Model.")
cwd = os.getcwd()
script = "train_nominal_model.py"

environments = [
    "MBHalfCheetah-v0",
    "PendulumSwingUp-v0",
    "MBHopper-v0",
    "MBWalker2d-v0",
]

commands = make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0, "hallucinate": False},
    common_hyper_args={"seed": [0], "protagonist-name": ["SAC", "MPO"]},
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
        "base-agent": ["SAC", "MPO", "BPTT"],
        "num-steps": [1, 4, 8],
    },
)

for cmd in commands:
    print(cmd)
