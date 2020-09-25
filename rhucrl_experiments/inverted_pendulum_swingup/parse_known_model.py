"""Python Script Template."""
import os
from itertools import product

import pandas as pd

STATISTICS = "statistics.json"


def parse_name(name: str):
    """Parse run name."""
    if name.startswith("Nominal"):
        alpha = 0.0
        robust = None
        wrapper = None
    else:
        alpha = float(name.split(" ")[-1].split("_")[0])
        robust = name.startswith("Correct")
        wrapper = name.split(" ")[-2]

    return dict(alpha=alpha, robust=robust, wrapper=wrapper)


def extend_data_frame(df, name_dict):
    """Extend data frame with a name dictionary."""
    df["counter"] = range(len(df))
    for key, value in name_dict.items():
        df[key] = value


def extend_nominal(df):
    """Extend nominal data frame."""
    wrappers = filter(lambda x: x is not None, df.wrapper.unique())
    robust_flags = filter(lambda x: x is not None, df.robust.unique())

    new_df = pd.DataFrame()
    for wrapper, robust in product(wrappers, robust_flags):
        new_df_ = df[df.alpha == 0.0].copy()
        new_df_["wrapper"] = wrapper
        new_df_["robust"] = robust
        new_df = pd.concat([new_df, new_df_])

    df = pd.concat([df, new_df])
    return df


def get_all_data_frames(base_dir="runs/RARLAgent/"):
    """Get experiment data frames."""
    if base_dir[-1] != "/":
        base_dir = base_dir + "/"

    df = pd.DataFrame()

    for run in os.listdir(base_dir):
        if run == ".DS_Store":
            continue
        run_info = parse_name(run)
        run_dir = f"{base_dir}/{run}"

        try:
            df_ = pd.read_json(f"{run_dir}/{STATISTICS}")
            # df_ = df_[df_.eval_return == df_.eval_return.max()]
        except ValueError:
            continue
        extend_data_frame(df_, run_info)

        df = pd.concat([df, df_])

    df = extend_nominal(df)

    return df


df = get_all_data_frames("runs/AdversarialMPCAgent")
df.to_pickle("pendulum_swing_up.pkl")
