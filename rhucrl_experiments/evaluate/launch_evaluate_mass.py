"""Run from rhucrl_experiments.evaluate folder."""
import socket

from lsf_runner import init_runner, make_commands

from rhucrl_experiments.evaluate.utilities import ENVIRONMENTS

RARL_DIR = "../../runs/RARLAgent"
ZERO_SUM_DIR = "../../runs/ZeroSumAgent"
SCRIPT = "evaluate_mass_change.py"
EXPERIMENTS = {
    "supermodularity": {"algorithm": "RARL_MF", "base-dir": RARL_DIR},
    "shallow": {"algorithm": "RHUCRL", "base-dir": ZERO_SUM_DIR},
    "greedy": {"algorithm": "RHUCRL", "base-dir": ZERO_SUM_DIR},
    "lazy": {"algorithm": "HUCRL", "base-dir": RARL_DIR},
}.get(socket.gethostname(), {"algorithm": "RARL", "base-dir": RARL_DIR})

runner = init_runner("EvaluateMassChange.", num_threads=4)
for seed in [0, 1, 2, 3, 4]:
    base_args = {"num-runs": 10, "seed": seed}
    base_args.update(**EXPERIMENTS)
    commands = make_commands(
        SCRIPT, base_args=base_args, common_hyper_args={"environment": ENVIRONMENTS}
    )
    runner.run_batch(commands)
