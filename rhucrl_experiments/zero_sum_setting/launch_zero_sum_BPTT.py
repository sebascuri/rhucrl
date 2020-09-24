"""Python Script Template."""

import os
import time

from lsf_runner import init_runner, make_commands

from rhucrl_experiments.zero_sum_setting.get_experiments import get_experiments

runner = init_runner("ZeroSum-BPTT", num_threads=2)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_zero_sum.py"

experiments = get_experiments()

for experiment in experiments:
    # MODEL-BASED + Hallucination + weak/strong
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"agent": "ZeroSum"},
        common_hyper_args={
            "seed": [0],  # , 1, 2, 3, 4
            "environment": experiment["environment"],
            "protagonist-name": ["BPTT"],
            "alpha": experiment["alpha"],
            "adversarial-wrapper": experiment["wrapper"],
            "hallucinate": [True],
            "strong-antagonist": [True, False],
        },
    )
    runner.run_batch(commands)
    time.sleep(2)

    # MODEL-BASED + Non-Hallucination.
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"agent": "ZeroSum"},
        common_hyper_args={
            "seed": [0],  # , 1, 2, 3, 4
            "environment": experiment["environment"],
            "protagonist-name": ["BPTT"],
            "alpha": experiment["alpha"],
            "adversarial-wrapper": experiment["wrapper"],
            "hallucinate": [False],
            "strong-antagonist": [False],
        },
    )
    runner.run_batch(commands)
    time.sleep(2)
