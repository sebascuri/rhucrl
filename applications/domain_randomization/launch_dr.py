import os

from lsf_runner import init_runner, make_commands


cwd = os.path.dirname(os.path.realpath(__file__))
script = "run_dr.py"

SEEDS = [0, 1, 2]
ALPHAS = [1.0, 2.0, 5.0, 10.0]
ENVS = ["half_cheetah", "hopper", "inverted_pendulum", "reacher", "swimmer", "walker"]
ENVS = [f"config/envs/{env}.yaml" for env in ENVS]
AGENTS = {
    "EPOPT": ["epopt"],
    "SAC": ["sac"],
    "PPO": ["ppo"],
    "BPTT": ["hucrl_a"],
    "MVE": ["hucrl_a"],
    "DataAugmentation": ["hucrl_a", "hucrl_b", "hucrl_c"],
    # "MaxiMin": ["mve"],
    # "BestResponse": ["mve"],
    # "RHUCRL": ["mve", "mve3", "da"],
}

for agent, agent_configs in AGENTS.items():
    runner = init_runner(f"AdversarialRL_{agent}", wall_time=24 * 60, num_threads=2)
    commands = []
    for agent_config in agent_configs:
        commands += make_commands(
            script,
            base_args={
                "agent": agent,
                "agent-config": f"config/agents/{agent_config}.yaml",
                "hallucinate": "hucrl" in agent_config,
            },
            common_hyper_args={
                "seed": SEEDS,
                "env-config": ENVS,
            },
        )
    runner.run(commands)