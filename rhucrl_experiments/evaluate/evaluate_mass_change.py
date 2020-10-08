"""Python Script Template."""
import os
from argparse import Namespace

import mujoco_py
import numpy as np
import pandas as pd

from rhucrl_experiments.evaluate.utilities import (
    get_command_line_parser,
    init_experiment,
)
from rhucrl_experiments.run import evaluate

eval_args = get_command_line_parser().parse_args()

list_dir = [*os.listdir(eval_args.base_dir)]
df = pd.DataFrame()
for path in list_dir:
    if eval_args.environment not in path:
        continue

    print(path)
    agent_dir = os.path.join(eval_args.base_dir, path)
    try:
        hparams = pd.read_json(os.path.join(agent_dir, "hparams.json"))
    except ValueError:
        continue
    args = Namespace(**hparams.iloc[0].to_dict())
    protagonist, environment = init_experiment(
        args, agent_dir, seed=eval_args.seed, eval_episodes=eval_args.num_runs
    )
    try:
        idx = environment.env.model.body_names.index("torso")
    except ValueError:
        idx = environment.env.model.body_names.index("r_forearm_link")
    original_mass = environment.env.model.body_mass[idx]

    for relative_mass in np.logspace(-2, 2, 11):
        environment.env.model.body_mass[idx] = original_mass * relative_mass

        try:
            evaluate(agent=protagonist, environment=environment, args=args)
            returns = protagonist.logger.get("eval_return")
        except mujoco_py.builder.MujocoException:
            returns = [-np.inf for _ in range(eval_args.num_runs)]

        df_ = hparams.copy()
        for i in range(eval_args.num_runs):
            df_[f"run_{i}"] = returns[-i]
        df_["run_min"] = np.min(returns[-eval_args.num_runs :])
        df_["run_max"] = np.max(returns[-eval_args.num_runs :])
        df_["run_mean"] = np.mean(returns[-eval_args.num_runs :])
        df_["run_std"] = np.std(returns[-eval_args.num_runs :])
        df_["mass"] = original_mass * relative_mass
        df_["relative_mass"] = relative_mass

        df = df.append(df_)

df.to_pickle(
    f"eval_mass_{eval_args.algorithm}_{eval_args.environment}_{eval_args.seed}.pickle"
)
