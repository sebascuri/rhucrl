"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

from rhucrl_experiments.get_environment import get_environment

runner = init_runner("ZeroSum-MF", num_threads=2)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_zero_sum.py"
MUJOCO_ENVIRONMENTS = get_environment().copy()
if "PendulumSwingUp-v0" in MUJOCO_ENVIRONMENTS:
    MUJOCO_ENVIRONMENTS.remove("PendulumSwingUp-v0")

AGENTS = ["TD3", "SAC", "MPO", "PPO", "VMPO"]
EXPERIMENT = [
    {
        "environment": get_environment(),
        "alpha": [0.01, 0.05, 0.1, 0.15, 0.2],
        "wrapper": ["noisy_action", "probabilistic_action"],
    },
    {
        "environment": MUJOCO_ENVIRONMENTS,
        "alpha": [1.0, 5.0, 10.0],
        "wrapper": ["external_force"],
    },
    {
        "environment": ["PendulumSwingUp-v0"],
        "alpha": [0.01, 0.05, 0.1, 0.15, 0.2],
        "wrapper": ["adversarial_pendulum"],
    },
]

for experiment in EXPERIMENT:
    # MODEL-FREE
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"agent": "RARL"},
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
