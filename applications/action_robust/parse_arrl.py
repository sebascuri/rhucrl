"""Parse Action Robust RL experiments."""
import os

import pandas as pd


class Experiment(object):
    """Experiment class to parse."""

    def __init__(self, file_name):
        splits = file_name[:-5].split("_")
        self.environment = splits[0][2:-3]
        self.kind = splits[1]
        self.alpha = float(splits[2])
        self.agent = splits[3]
        if len(splits) == 7:
            self.config = splits[4] + "_" + splits[5]
        else:
            self.config = splits[4]

        self.seed = int(splits[-1])

        df = pd.read_json(file_name)
        self.train_returns = df.train_return

        robust_file_name = file_name[:-5] + "_robust.json"
        df = pd.read_json(robust_file_name)
        self.robust_returns = df.train_return

    def get_df(self):
        """Get the experiment as a data frame."""
        return pd.DataFrame(vars(self))


def parse_dir(path=None):
    """Parse all experiments in directory."""
    if path is None:
        path = os.getcwd()
    df = pd.DataFrame()
    for file_name in filter(
        lambda x: x.endswith(".json") and not x.endswith("robust.json"),
        os.listdir(path),
    ):
        try:
            experiment = Experiment(file_name)
            df = df.append(experiment.get_df())
        except ValueError:
            pass
    return df


if __name__ == "__main__":
    import socket

    df = parse_dir()
    df.reset_index(inplace=True)
    df.to_json(f"action_robust_{socket.gethostname()}.json")
