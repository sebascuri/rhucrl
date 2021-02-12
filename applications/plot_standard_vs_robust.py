"""Python Script Template."""
import matplotlib.pyplot as plt
import pandas as pd

from applications.plotters import COLORS, LABELS, LINESTYLE, set_figure_params

df = pd.read_json("adversarial_robust.json")
group = df.groupby(["environment", "agent", "config", "alpha"])
# returns = group.train_returns

returns = group.robust_returns
envs = df.environment.unique()


def plot_adversarial_rhucrl(
    ax, alpha, train_returns, environment, config, beta=0.3, agent="RHUCRL", label=None
):
    """Plot RHUCRL."""
    if label is None:
        label = agent

    ax.plot(
        alpha,
        train_returns.max()[environment][agent][config],
        color=COLORS[label],
        label=LABELS[label],
        linestyle=LINESTYLE[label],
    )
    ax.fill_between(
        alpha,
        train_returns.max()[environment][agent][config]
        - beta * train_returns.std()[environment][agent][config],
        train_returns.max()[environment][agent][config]
        + beta * train_returns.std()[environment][agent][config],
        alpha=0.3,
        color=COLORS[label],
    )


def plot_adversarial_robust(
    ax, alpha, train_returns, environment, agent, config, beta=0.3, label=None
):
    """Plot algorithms."""
    if label is None:
        label = agent
    # ax.plot(
    #     alpha,
    #     train_returns.max()[environment][agent][config],
    #     color=COLORS[label],
    #     label=LABELS[label],
    #     linestyle=LINESTYLE[label],
    # )
    # ax.fill_between(
    #     alpha,
    #     train_returns.max()[environment][agent][config]
    #     - beta * train_returns.std()[environment][agent][config],
    #     train_returns.max()[environment][agent][config]
    #     + beta * train_returns.std()[environment][agent][config],
    #     alpha=0.3,
    #     color=COLORS[label],
    # )

    ax.plot(
        alpha,
        train_returns.mean()[environment][agent][config],
        color=COLORS[label],
        label=LABELS[label],
        linestyle=LINESTYLE[label],
    )
    ax.fill_between(
        alpha,
        train_returns.mean()[environment][agent][config]
        - beta * train_returns.std()[environment][agent][config],
        train_returns.mean()[environment][agent][config]
        + beta * train_returns.std()[environment][agent][config],
        alpha=0.3,
        color=COLORS[label],
    )


def plot_standard_robust(
    ax, alpha, train_returns, environment, agent, config, beta=0.3, label=None
):
    """Plot standard."""
    if label is None:
        label = agent
    ax.plot(
        alpha,
        train_returns.max()[environment][agent][config],
        color=COLORS[label],
        label=LABELS[label],
        linestyle=LINESTYLE[label],
    )
    ax.fill_between(
        alpha,
        train_returns.max()[environment][agent][config]
        - beta * train_returns.std()[environment][agent][config],
        train_returns.max()[environment][agent][config]
        + beta * train_returns.std()[environment][agent][config],
        alpha=0.3,
        color=COLORS[label],
    )


set_figure_params(serif=True, fontsize=10)
fig, axes = plt.subplots(ncols=2, nrows=1, sharex="all", sharey="all")
fig.set_size_inches(6.5 / 2, 1.5)

# %% Half Cheetah
env = "HalfCheetah"
ax = axes[0]
returns = group.train_returns
alpha = group.alpha.mean()[env].unique()
plot_adversarial_rhucrl(ax, alpha, returns, env, "hucrl_c")
plot_standard_robust(ax, alpha, returns, env, "BestResponse", "hucrl_c")
plot_standard_robust(ax, alpha, returns, env, "MaxiMin", "hucrl_c")
plot_standard_robust(ax, alpha, returns, env, "MaxiMin", "sac", label="MaxiMinMF")
plot_standard_robust(ax, alpha, returns, env, "BPTT", "hucrl_c")
plot_standard_robust(ax, alpha, returns, env, "RARL", "ppo")
plot_standard_robust(ax, alpha, returns, env, "RAP", "ppo")

ax.set_title("Non-Robust Return")

# %% Hopper
env = "HalfCheetah"
ax = axes[1]
returns = group.robust_returns

alpha = group.alpha.mean()[env].unique()
plot_adversarial_rhucrl(ax, alpha, returns, env, "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "BestResponse", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "sac", label="MaxiMinMF")
plot_adversarial_robust(ax, alpha, returns, env, "BPTT", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "RARL", "ppo")
plot_adversarial_robust(ax, alpha, returns, env, "RAP", "ppo")

ax.set_title("Robust Return")


handles, labels = axes[0].get_legend_handles_labels()

axes[0].legend(
    handles=handles[3:],
    labels=labels[3:],
    loc="upper left",
    bbox_to_anchor=(0.0, 0.4),
    frameon=False,
    ncol=2,
    handlelength=1.5,
    columnspacing=0.4,
    handletextpad=0.1,
)

axes[1].legend(
    handles=handles[:3],
    labels=labels[:3],
    loc="upper left",
    bbox_to_anchor=(0.1, 1.05),
    frameon=False,
    handlelength=1.5,
    handletextpad=0.1,
)

axes[0].set_xticklabels([])
axes[1].set_xticklabels([])

# axes[0].legend(
#     handles=handles[2:4],
#     labels=labels[2:4],
#     loc="upper left",
#     bbox_to_anchor=(0.5, 0.4),
#     frameon=False,
# )

# axes[0].set_xlabel("Input Magnitude")
fig.text(0.4, 0.05, "Input Magnitude", va="center", fontsize=12)

plt.tight_layout(h_pad=0.5, w_pad=0.2)
plt.savefig("adversarial_robust_vs_standard.pdf")

# plt.show()
