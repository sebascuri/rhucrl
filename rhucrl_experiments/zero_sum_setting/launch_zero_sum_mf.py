"""Python Script Template."""
import os
import time

from lsf_runner import init_runner, make_commands

from rhucrl_experiments.zero_sum_setting.get_experiments import get_experiments

runner = init_runner("ZeroSum-MF", num_threads=2)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_zero_sum.py"

AGENTS = ["TD3", "SAC", "MPO", "PPO", "VMPO"]
experiments = get_experiments()

for experiment in experiments:
    # MODEL-FREE
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"agent": "ZeroSum"},
        common_hyper_args={
            "seed": [0],  # , 1, 2, 3, 4
            "environment": experiment["environment"],
            "protagonist-name": AGENTS,
            "alpha": experiment["alpha"],
            "adversarial-wrapper": experiment["wrapper"],
            "hallucinate": [False],
        },
    )
    runner.run_batch(commands)
    time.sleep(2)
