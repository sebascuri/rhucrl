"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

# import time


runner = init_runner("RARL", wall_time=24 * 60, num_threads=4)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_rarl.py"

# MODEL-BASED + Hallucination + weak/strong
commands = make_commands(
    f"{cwd}/{script}",
    base_args={"agent": "RARL"},
    common_hyper_args={
        "seed": [0, 1, 2],  # ,
        "environment": [
            "MBHalfCheetah-v0",
            "MBHopper-v0",
            "MBWalker2d-v0",
            "MBSwimmer-v0",
            "MBReacher3d-v0",
            "PendulumSwingUp-v0",
        ],
        "alpha": [0.1, 0.2, 0.3],
        "protagonist-name": ["VMPO", "TD3"],  # PPO
        "adversarial-wrapper": ["probabilistic_action", "noisy_action"],
        "hallucinate": [False],
        "strong-antagonist": [False],
    },
)
# print(len(commands))
runner.run_batch(commands)
# time.sleep(2)
