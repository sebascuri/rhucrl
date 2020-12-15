import os

from lsf_runner import init_runner, make_commands


runner = init_runner("AdversarialRL", wall_time=24 * 60, num_threads=2)
cwd = os.path.dirname(os.path.realpath(__file__))
script = "run.py"

SEEDS = [0, 1, 2]
ALPHAS = [1.0, 2.0, 5.0, 10.0]
ENVS = ["half_cheetah", "hopper", "inverted_pendulum", "reacher", "swimmer", "walker"]
ENVS = [f"config/envs/{env}.yaml" for env in ENVS]
AGENTS = {
    "SAC": ["sac"],
    "PPO": ["ppo"],
    "MVE": ["hucrl", "hucrl3"],
    "DataAugmentation": ["hucrl", "hucrl3"],
    "RAP": ["ppo", "sac"],
    "RARL": ["ppo", "sac"],
    "HRARL": ["hucrl"],
    "MaxiMin": ["mve"],
    "BestResponse": ["mve"],
    "RHUCRL": ["mve", "mve3", "da"],
}

commands = []
for agent, agent_configs in AGENTS.items():
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
