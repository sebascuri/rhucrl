"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

# import time


runner = init_runner("RHUCRL", num_threads=1)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_zero_sum.py"

# MODEL-BASED + Hallucination + weak/strong
commands = make_commands(
    f"{cwd}/{script}",
    base_args={"agent": "ZeroSum"},
    common_hyper_args={
        "seed": [0],  # ,
        "environment": [
            "MBHalfCheetah-v0",
            "MBHopper-v0",
            "MBWalker2d-v0",
            "MBSwimmer-v0",
            "MBReacher3d-v0",
        ],
        "alpha": [0.1, 0.2, 0.3],
        "protagonist-name": ["BPTT"],
        "adversarial-wrapper": ["probabilistic_action", "noisy_action"],
        "hallucinate": [True],
        "strong-antagonist": [True],
    },
)

commands += make_commands(
    f"{cwd}/{script}",
    base_args={"agent": "ZeroSum"},
    common_hyper_args={
        "seed": [0],  # ,
        "environment": [
            "MBHalfCheetah-v0",
            "MBHopper-v0",
            "MBWalker2d-v0",
            "MBSwimmer-v0",
            "MBReacher3d-v0",
        ],
        "alpha": [0.1, 0.2, 0.3],
        "protagonist-name": ["MVE"],
        "adversarial-wrapper": ["probabilistic_action", "noisy_action"],
        "base-agent": ["TD3", "BPTT"],
        "hallucinate": [True],
        "strong-antagonist": [True],
    },
)
runner.run_batch(commands)
# print(len(commands))
