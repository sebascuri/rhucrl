"""Parse inverted pendulum results."""

import pandas as pd

alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
for robustness in ["adversarial", "action"]:
    df = pd.DataFrame()
    for agent in ["HUCRL", "RHUCRL"]:
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
    df.to_json(f"{robustness}.json")
