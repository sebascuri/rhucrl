"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

# import time


runner = init_runner("RARL", num_threads=1)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_rarl.py"

# MODEL-BASED + Hallucination + weak/strong
commands = make_commands(
    f"{cwd}/{script}",
    base_args={"agent": "RARL"},
    common_hyper_args={
        "seed": [1, 2],  # ,
        "environment": [
            "MBHalfCheetah-v0",
            "MBHopper-v0",
            "MBWalker2d-v0",
            "MBSwimmer-v0",
            "MBReacher3d-v0",
            "PendulumSwingUp-v0",
        ],
        "alpha": [1.0, 5.0, 10.0],
        "protagonist-name": ["PPO", "VMPO", "TD3"],
        "adversarial-wrapper": ["external_force"],
        "hallucinate": [False],
        "strong-antagonist": [False],
    },
)
# print(len(commands))
runner.run_batch(commands)
# time.sleep(2)
