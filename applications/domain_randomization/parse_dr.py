"""Parse Action Robust RL experiments."""
import os

import pandas as pd


class Experiment(object):
    """Experiment class to parse."""

    def __init__(self, file_name):
        splits = file_name[:-5].split("_")
        self.environment = splits[0][2:-3]
        self.agent = splits[1]
        if len(splits) == 6:
            self.config = splits[2] + "_" + splits[3]
        else:
            self.config = splits[2]
        self.seed = int(splits[-2])

        df = pd.read_json(file_name)
        self.train_returns = df.train_return.to_numpy()
        self.mass = df["mass-change-0"].to_numpy()

    def get_df(self):
        """Get the experiment as a data frame."""
        return pd.DataFrame(vars(self))


def parse_dir(path=None):
    """Parse all experiments in directory."""
    if path is None:
        path = os.getcwd()
    df = pd.DataFrame()
    for file_name in filter(lambda x: x.endswith("eval.json"), os.listdir(path)):
        experiment = Experiment(file_name)
        df = df.append(experiment.get_df())
    return df


if __name__ == "__main__":
    import socket

    df = parse_dir(path=os.path.dirname(os.path.realpath(__file__)))
    df.reset_index(inplace=True)
    df.to_json(f"parameter_robust_{socket.gethostname()}.json")
