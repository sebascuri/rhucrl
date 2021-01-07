"""Parse Adversarial RL experiments."""
import pandas as pd
import os


class Experiment(object):
    """Experiment class to parse."""

    def __init__(self, file_name):
        splits = file_name[:-5].split("_")
        self.environment = splits[0][2:-3]
        self.alpha = float(splits[1])
        self.agent = splits[2]
        self.config = splits[3]
        self.seed = int(splits[4])

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
    for file_name in os.listdir(path):
        if file_name.endswith(".json") and not file_name.endswith("robust.json"):
            experiment = Experiment(file_name)
            df = df.append(experiment.get_df())
    return df


if __name__ == "__main__":
    df = parse_dir()
    df.to_json("adversarial_rl.json")
