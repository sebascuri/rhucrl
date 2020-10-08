"""Python Script Template."""
import json
import os
from itertools import product

import pandas as pd

STATISTICS = "statistics.json"
HPARAMS = "hparams.json"


def parse_name(name: str, hyper_params: dict = None):
    """Parse run name."""
    if name.startswith("Nominal") and hyper_params is None:
        alpha = 0.0
        algorithm = None
        wrapper = None
    else:
        if hyper_params is None:
            alpha = float(name.split(" ")[-1].split("_")[0])
            if name.startswith("Correct"):
                algorithm = "RHUCRL"
            else:
                algorithm = "HUCRL"
            wrapper = name.split(" ")[-2]
        else:
            if hyper_params["agent"] == "RARL":
                algorithm = "RARL"
            else:
                if name.startswith("Correct"):
                    algorithm = "RHUCRL"
                else:
                    algorithm = "HUCRL"
            alpha = hyper_params["alpha"]
            wrapper = hyper_params["adversarial_wrapper"]

    return dict(alpha=alpha, algorithm=algorithm, wrapper=wrapper)


def extend_data_frame(df, name_dict):
    """Extend data frame with a name dictionary."""
    df["counter"] = range(len(df))
    for key, value in name_dict.items():
        df[key] = value


def extend_nominal(df):
    """Extend nominal data frame."""
    wrappers = filter(lambda x: x is not None, df.wrapper.unique())
    algorithms = ["HUCRL", "RHUCRL"]

    new_df = pd.DataFrame()
    for wrapper, algorithm in product(wrappers, algorithms):
        new_df_ = df[df.alpha == 0.0].copy()
        new_df_["wrapper"] = wrapper
        new_df_["algorithm"] = algorithm
        new_df = pd.concat([new_df, new_df_])

    df = pd.concat([df, new_df])
    return df


def get_all_data_frames(base_dir="runs/RARLAgent/", extend=False):
    """Get experiment data frames."""
    if base_dir[-1] != "/":
        base_dir = base_dir + "/"

    df = pd.DataFrame()

    for run in os.listdir(base_dir):
        if run == ".DS_Store":
            continue
        run_dir = f"{base_dir}/{run}"
        try:
            with open(f"{run_dir}/{HPARAMS}", "r") as f:
                hyper_params = json.load(f)
        except OSError:
            hyper_params = None

        run_info = parse_name(run, hyper_params)

        try:
            df_ = pd.read_json(f"{run_dir}/{STATISTICS}")
            # df_ = df_[df_.eval_return == df_.eval_return.max()]
        except ValueError:
            continue
        extend_data_frame(df_, run_info)

        df = pd.concat([df, df_])

    if extend:
        df = extend_nominal(df)

    return df


df = pd.concat(
    [
        get_all_data_frames("runs/AdversarialMPCAgent", extend=True),
        get_all_data_frames("runs/RARLAgent", extend=False),
    ]
)

df.to_pickle("pendulum_swing_up.pkl")
