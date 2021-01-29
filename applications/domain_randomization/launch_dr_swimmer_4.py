"""Launch Domain Randomization Experiments."""
import os

from lsf_runner import init_runner, make_commands

cwd = os.path.dirname(os.path.realpath(__file__))
script = "run_dr.py"

SEEDS = [0, 1, 2]
ENVS = ["swimmer"]

ENVS = [f"config/envs/{env}.yaml" for env in ENVS]
AGENTS = {"EPOPT": ["epopt"], "DomainRandomization": ["epopt_dr"], "PPO": ["ppo"]}

for agent, agent_configs in AGENTS.items():
    runner = init_runner(
        f"DomainRandomizationRL_{agent}", wall_time=4 * 60, num_threads=1, memory=4096
    )
    commands = []
    for agent_config in agent_configs:
        commands += make_commands(
            script,
            base_args={
                "agent": agent,
                "agent-config": f"config/agents/{agent_config}.yaml",
            },
            common_hyper_args={"seed": SEEDS, "env-config": ENVS},
        )
    runner.run_batch(commands)
