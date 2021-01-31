"""Parse inverted pendulum results."""

import pandas as pd

alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
agents = ["HUCRL", "RHUCRL", "baseline"]

for robustness in ["adversarial", "action"]:
    df = pd.DataFrame()
    for agent in agents:
        mean, std, min_, max_ = [], [], [], []
        for alpha in alphas:
            df_ = pd.read_json(f"{agent}_{robustness}_{alpha}.json")
            mean.append(df_.eval_return.mean())
            std.append(df_.eval_return.std())
            min_.append(df_.eval_return.min())
            max_.append(df_.eval_return.max())
        df_ = pd.DataFrame(
            {
                "mean": mean,
                "std": std,
                "min": min_,
                "max": max_,
                "alpha": alphas,
                "agent": agent,
            }
        )
        df = df.append(df_)
    df.reset_index(inplace=True)
    df.to_json(f"{robustness}_robust.json")

df = pd.DataFrame()
for agent in agents:
    df_ = pd.read_json(f"{agent}_parameter.json")
    df_ = df_[df_.mass.notna()]
    mass = df_.mass.unique()
    group = df_.groupby("mass").eval_return
    df_ = pd.DataFrame(
        {
            "mean": group.mean(),
            "std": group.std(),
            "min": group.min(),
            "max": group.max(),
            "alpha": mass,
            "agent": agent,
        }
    )
    df = df.append(df_)
df.reset_index(inplace=True)
df.to_json(f"parameter_robust.json")
