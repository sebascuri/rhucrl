"""Python Script Template."""
from lsf_runner import init_runner, make_commands

runner = init_runner(
    "Adversarial-HalfCheetah", num_threads=1, use_gpu=False, wall_time=24 * 60
)
cmd_list = make_commands(
    "run_half_cheetah.py",
    base_args={"train-episodes": 100},
    common_hyper_args={
        "agent": ["RARL", "Zero-Sum"],
        "protagonist-name": ["SAC", "TD3", "MPO"],
        "alpha": [1.0, 5.0, 10.0],
    },
)
runner.run_batch(cmd_list)
