"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

# import time


runner = init_runner("HUCRL", num_threads=1)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_rarl.py"

# MODEL-BASED + Hallucination + weak/strong
commands = make_commands(
    f"{cwd}/{script}",
    base_args={"agent": "RARL", "train-episodes": 500, "train-antagonist-episodes": 0},
    common_hyper_args={
        "seed": [0],  # ,
        "environment": [
            "MBHalfCheetah-v0",
            "MBHopper-v0",
            "MBWalker2d-v0",
            "MBSwimmer-v0",
            "MBReacher3d-v0",
        ],
        "alpha": [0.0],
        "protagonist-name": ["MVE"],
        "adversarial-wrapper": ["external_force"],
        "base-agent": ["TD3", "BPTT", "MPO"],
        "hallucinate": [True],
        "strong-antagonist": [True],
        "train-antagonist-episodes": [0],
    },
)
# print(len(commands))
runner.run_batch(commands)
# time.sleep(2)


commands = make_commands(
    f"{cwd}/{script}",
    base_args={"agent": "RARL", "train-episodes": 500, "train-antagonist-episodes": 0},
    common_hyper_args={
        "seed": [0],  # ,
        "environment": [
            "MBHalfCheetah-v0",
            "MBHopper-v0",
            "MBWalker2d-v0",
            "MBSwimmer-v0",
            "MBReacher3d-v0",
        ],
        "alpha": [0.0],
        "protagonist-name": ["BPTT"],
        "adversarial-wrapper": ["external_force"],
        "hallucinate": [True],
        "strong-antagonist": [True],
    },
)
runner.run_batch(commands)
