"""Launch RHUCRL for adversarial RL."""
import os

from lsf_runner import init_runner, make_commands

cwd = os.path.dirname(os.path.realpath(__file__))
script = "run_adv_rl.py"

SEEDS = [0]
ALPHAS = [1.0, 2.0, 5.0, 10.0]
ENVS = ["half_cheetah", "reacher"]
ENVS = [f"config/envs/{env}.yaml" for env in ENVS]

AGENTS = {"RHUCRL": ["hucrl_a", "hucrl_b", "hucrl_c"]}

for agent, agent_configs in AGENTS.items():
    runner = init_runner(
        f"AdversarialRL_{agent}", wall_time=2 * 24 * 60, num_threads=1, memory=4096
    )
    commands = []
    for agent_config in agent_configs:
        commands += make_commands(
            script,
            base_args={
                "agent": agent,
                "agent-config": f"config/agents/{agent_config}.yaml",
            },
            common_hyper_args={"seed": SEEDS, "env-config": ENVS, "alpha": ALPHAS},
        )
    runner.run_batch(commands)
