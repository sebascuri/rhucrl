"""Python Script Template."""
import os

import pandas as pd

keys = ["action", "adversarial", "parameter"]

for key in keys:
    df = pd.DataFrame()
    for file_name in filter(
        lambda x: x.startswith(key) and len(x.split("_")) == 3 and x.endswith(".json"),
        os.listdir(),
    ):
        df_ = pd.read_json(file_name)
        df = df.append(df_)
    df.reset_index(inplace=True)
    df.to_json(f"{key}_robust.json")
