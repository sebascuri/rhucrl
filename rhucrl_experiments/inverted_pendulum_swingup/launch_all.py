"""Python Script Template."""
import os

from lsf_runner import init_runner, make_commands

runner = init_runner("Known Model.")
cwd = os.getcwd()
script = "run_known_model.py"

# commands = make_commands(
#     f"{cwd}/{script}",
#     base_args={"max-steps": 500, "alpha": 0},
#     common_hyper_args={"seed": [0, 1, 2, 3, 4]},
# )

for seed in [0, 1, 2, 3, 4]:
    print(seed)
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"max-steps": 500, "strong-antagonist": True},
        common_hyper_args={
            "alpha": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],  # [],
            "seed": [seed],
            "nominal-model": [True, False],
            "adversarial-wrapper": [
                "noisy_action",
                "probabilistic_action",
                # "adversarial_pendulum",
            ],
        },
    )

    runner.run_batch(commands)
