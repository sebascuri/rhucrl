"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

runner = init_runner("RARL", wall_time=24 * 60, num_threads=2)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "train_rarl.py"

AGENTS = ["TD3"]
EXPERIMENT = [
    {
        "environment": ["MBHalfCheetah-v0", "MBHopper-v0", "MBWalker2d-v0"],
        "alpha": [0.01, 0.05, 0.1, 0.15, 0.2],
        "wrapper": ["noisy_action", "probabilistic_action"],
    },
    {
        "environment": ["MBHalfCheetah-v0", "MBHopper-v0", "MBWalker2d-v0"],
        "alpha": [1.0, 5.0, 10.0],
        "wrapper": ["external_force"],
    },
    {
        "environment": ["PendulumSwingUp-v0"],
        "alpha": [0.01, 0.05, 0.1, 0.15, 0.2],
        "wrapper": ["noisy_action", "probabilistic_action", "adversarial_pendulum"],
    },
]

command_count = []
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
    command_count += commands
    # runner.run_batch(commands)

    # MODEL-BASED + Hallucination + weak/strong
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"seed": 0},
        common_hyper_args={
            "environment": experiment["environment"],
            "protagonist-name": ["BPTT"],
            "alpha": experiment["alpha"],
            "adversarial-wrapper": experiment["wrapper"],
            "hallucinate": [True],
            "strong-antagonist": [True, False],
            "num-steps": [1, 4],
        },
    )
    command_count += commands
    # runner.run_batch(commands)

    # MODEL-BASED + Non-Hallucination.
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"seed": [0]},
        common_hyper_args={
            "environment": experiment["environment"],
            "protagonist-name": ["BPTT"],
            "alpha": experiment["alpha"],
            "adversarial-wrapper": experiment["wrapper"],
            "hallucinate": [False],
            "strong-antagonist": [False],
            "num-steps": [1],
        },
    )
    command_count += commands
    # runner.run_batch(commands)

    # MODEL-Augmented + Hallucination.
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"seed": 0},
        common_hyper_args={
            "environment": experiment["environment"],
            "protagonist-name": ["MVE"],
            "alpha": experiment["alpha"],
            "adversarial-wrapper": experiment["wrapper"],
            "hallucinate": [True],
            "strong-antagonist": [True, False],
            "num-steps": [1, 4],
            "base-agent-name": AGENTS + ["BPTT"],
        },
    )
    command_count += commands
    # runner.run_batch(commands)

    # MODEL-Augmented + Non-Hallucination.
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"seed": 0},
        common_hyper_args={
            "environment": experiment["environment"],
            "protagonist-name": ["MVE"],
            "alpha": experiment["alpha"],
            "adversarial-wrapper": experiment["wrapper"],
            "hallucinate": [False],
            "strong-antagonist": [False],
            "num-steps": [1],
            "base-agent-name": AGENTS + ["BPTT"],
        },
    )
    command_count += commands
    # runner.run_batch(commands)

print(len(command_count))
