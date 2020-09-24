"""Python Script Template."""
import os
import time

from lsf_runner import init_runner, make_commands

from rhucrl_experiments.rarl_setting.get_experiments import get_experiments

runner = init_runner("RARL-MA", num_threads=2)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_rarl.py"

AGENTS = ["TD3", "BPTT"]
experiments = get_experiments()

for experiment in experiments:
    # MODEL-Augmented + Hallucination.
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"agent": "RARL"},
        common_hyper_args={
            "seed": [0],  # , 1, 2, 3, 4
            "environment": experiment["environment"],
            "protagonist-name": ["MVE"],
            "alpha": experiment["alpha"],
            "adversarial-wrapper": experiment["wrapper"],
            "hallucinate": [True],
            "strong-antagonist": [True, False],
            "num-steps": [1],
            "base-agent": AGENTS,
        },
    )
    runner.run_batch(commands)
    time.sleep(2)

    # MODEL-Augmented + Non-Hallucination.
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"agent": "RARL"},
        common_hyper_args={
            "seed": [0],  # , 1, 2, 3, 4
            "environment": experiment["environment"],
            "protagonist-name": ["MVE"],
            "alpha": experiment["alpha"],
            "adversarial-wrapper": experiment["wrapper"],
            "hallucinate": [False],
            "strong-antagonist": [False],
            "num-steps": [1],
            "base-agent-name": AGENTS,
        },
    )
    runner.run_batch(commands)
    time.sleep(2)
