"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

runner = init_runner("ZeroSum", wall_time=24 * 60, num_threads=2)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_zero_sum.py"

AGENTS = ["SAC"]
EXPERIMENT = [
    {
        "environment": ["MBHalfCheetah-v0"],
        "alpha": [0.1],
        "wrapper": ["noisy_action", "probabilistic_action"],
    }
]

for experiment in EXPERIMENT:
    # MODEL-FREE
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"seed": 0},
        common_hyper_args={
            "environment": experiment["environment"],
            "protagonist-name": AGENTS,
            "alpha": experiment["alpha"],
            "adversarial-wrapper": experiment["wrapper"],
            "hallucinate": [False],
        },
    )
    runner.run_batch(commands)
