"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

runner = init_runner("TrainKnownModel", wall_time=24 * 60, num_threads=4)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_nominal.py"

environments = [
    "MBHalfCheetah-v0",
    "PendulumSwingUp-v0",
    "MBHopper-v0",
    "MBWalker2d-v0",
]

AGENTS = ["SAC", "TD3", "DPG", "MPO", "PPO", "TRPO", "VMPO"]
commands = make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0, "hallucinate": False},
    common_hyper_args={"seed": [0], "protagonist-name": AGENTS},
)

commands += make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0},
    common_hyper_args={
        "seed": [0],
        "hallucinate": [True, False],
        "protagonist-name": ["BPTT"],
        "num-steps": [1, 4, 8],
    },
)

commands += make_commands(
    f"{cwd}/{script}",
    base_args={"alpha": 0, "protagonist-name": "MVE"},
    common_hyper_args={
        "seed": [0],
        "hallucinate": [True, False],
        "base-agent": AGENTS + ["BPTT"],
        "num-steps": [1, 4, 8],
    },
)

# print(len(commands))
runner.run(commands)
