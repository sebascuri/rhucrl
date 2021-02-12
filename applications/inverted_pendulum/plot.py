"""Python Script Template."""
import matplotlib.pyplot as plt
import pandas as pd

from applications.plotters import COLORS, LABELS, LINESTYLE, set_figure_params

set_figure_params(serif=True, fontsize=10)
fig, ax = plt.subplots(ncols=3, nrows=1, sharey="row")
fig.set_size_inches(6.5, 1.5)
df = pd.read_json("adversarial_robust.json")
alpha = df.alpha.unique()
agents = ["RHUCRL", "HUCRL", "baseline"]
for agent in agents:
    df_ = df[df.agent == agent].sort_values("alpha")
    mean, std = df_["mean"], df_["std"]
    ax[0].plot(
        alpha,
        mean,
        label=LABELS[agent],
        color=COLORS[agent],
        linestyle=LINESTYLE[agent],
    )
    ax[0].fill_between(
        alpha, mean - 3 * std, mean + 3 * std, alpha=0.3, color=COLORS[agent]
    )

# ax[0].legend(loc="best")
ax[0].set_xlabel("Input Magnitude")
ax[0].set_ylabel("Returns")
ax[0].set_title("Adversarial Robust")

df = pd.read_json("action_robust.json")
alpha = df.alpha.unique()
agents = ["RHUCRL", "HUCRL", "baseline"]
for agent in agents:
    df_ = df[df.agent == agent].sort_values("alpha")
    mean, std = df_["mean"], df_["std"]
    ax[1].plot(
        alpha,
        mean,
        label=LABELS[agent],
        color=COLORS[agent],
        linestyle=LINESTYLE[agent],
    )
    ax[1].fill_between(
        alpha, mean - 3 * std, mean + 3 * std, alpha=0.3, color=COLORS[agent]
    )

# ax[1].legend(loc="best")
ax[1].set_xlabel("Mixture Parameter")
ax[1].set_title("Action Robust")

df = pd.read_json("parameter_robust.json")
mass = df.alpha.unique()
agents = ["RHUCRL", "HUCRL", "baseline"]
for agent in agents:
    df_ = df[df.agent == agent].sort_values("mass")
    mean, std = df_["mean"], df_["std"]
    ax[2].plot(
        mass, mean, label=LABELS[agent], color=COLORS[agent], linestyle=LINESTYLE[agent]
    )
    ax[2].fill_between(
        mass, mean - 3 * std, mean + 3 * std, alpha=0.3, color=COLORS[agent]
    )

ax[2].set_xlabel("Relative Mass")
ax[2].set_title("Parameter Robust")

ax[0].legend(
    loc="upper left", bbox_to_anchor=(-0.03, 0.7), frameon=False, handlelength=1.5
)
plt.tight_layout(pad=0.0)
plt.savefig("pendulum_swing_up.pdf")
