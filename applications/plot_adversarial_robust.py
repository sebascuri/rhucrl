"""Python Script Template."""
import matplotlib.pyplot as plt
import pandas as pd

from applications.plotters import AXES, COLORS, LABELS, LINESTYLE, set_figure_params

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


set_figure_params(serif=True, fontsize=10)
fig, axes = plt.subplots(ncols=3, nrows=2, sharex="all")
fig.set_size_inches(6.5, 3.5)

# %% Half Cheetah
env = "HalfCheetah"
ax = axes[AXES[env]]

alpha = group.alpha.mean()[env].unique()
plot_adversarial_rhucrl(ax, alpha, returns, env, "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "BestResponse", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "sac", label="MaxiMinMF")
plot_adversarial_robust(ax, alpha, returns, env, "BPTT", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "RARL", "ppo")
plot_adversarial_robust(ax, alpha, returns, env, "RAP", "ppo")

ax.set_title(env)

# %% Hopper
env = "Hopper"
ax = axes[AXES[env]]

alpha = group.alpha.mean()[env].unique()
plot_adversarial_rhucrl(ax, alpha, returns, env, "hucrl_a")
plot_adversarial_robust(ax, alpha, returns, env, "BestResponse", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "hucrl_c")
plot_adversarial_robust(
    ax, alpha, returns, env, "MaxiMin", "hucrl_a", label="MaxiMinMF"
)

plot_adversarial_robust(ax, alpha, returns, env, "BPTT", "hucrl_a")

plot_adversarial_robust(ax, alpha, returns, env, "RARL", "ppo")
plot_adversarial_robust(ax, alpha, returns, env, "RAP", "ppo")

ax.set_title(env)

# %% Reacher2d
env = "Reacher2d"
ax = axes[AXES[env]]

alpha = group.alpha.mean()[env].unique()
plot_adversarial_rhucrl(ax, alpha, returns, env, "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "BestResponse", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "sac", label="MaxiMinMF")

plot_adversarial_robust(ax, alpha, returns, env, "BPTT", "hucrl_c")

# plot_adversarial_robust(ax, alpha, returns, env, "RARL", "ppo")
# plot_adversarial_robust(ax, alpha, returns, env, "RAP", "ppo")

ax.set_title(env)

# %% InvertedPendulum
env = "InvertedPendulum"
ax = axes[AXES[env]]

alpha = group.alpha.mean()[env].unique()
plot_adversarial_rhucrl(ax, alpha, returns, env, "hucrl_a")
plot_adversarial_robust(ax, alpha, returns, env, "BestResponse", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "sac", label="MaxiMinMF")

plot_adversarial_robust(ax, alpha, returns, env, "BPTT", "hucrl_c")

plot_adversarial_robust(ax, alpha, returns, env, "RARL", "ppo")
plot_adversarial_robust(ax, alpha, returns, env, "RAP", "ppo")

ax.set_title(env)

# %% Swimmer
env = "Swimmer"
ax = axes[AXES[env]]

alpha = group.alpha.mean()[env].unique()
plot_adversarial_rhucrl(ax, alpha, returns, env, "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "BestResponse", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "sac", label="MaxiMinMF")

plot_adversarial_robust(ax, alpha, returns, env, "BPTT", "hucrl_c")

plot_adversarial_robust(ax, alpha, returns, env, "RARL", "ppo")
plot_adversarial_robust(ax, alpha, returns, env, "RAP", "ppo")

ax.set_title(env)

# %% Walker2d
env = "Walker2d"
ax = axes[AXES[env]]

alpha = group.alpha.mean()[env].unique()
plot_adversarial_rhucrl(ax, alpha, returns, env, "hucrl_a")
plot_adversarial_robust(ax, alpha, returns, env, "BestResponse", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "hucrl_c")
plot_adversarial_robust(ax, alpha, returns, env, "MaxiMin", "sac", label="MaxiMinMF")

plot_adversarial_robust(ax, alpha, returns, env, "BPTT", "hucrl_c")

plot_adversarial_robust(ax, alpha, returns, env, "RARL", "ppo")
plot_adversarial_robust(ax, alpha, returns, env, "RAP", "ppo")

ax.set_title(env)

# %% Plot Formatting
axes[1, 1].set_xlabel("Input Magnitude")
fig.text(0.0, 0.56, "Robust Return", va="center", rotation="vertical", fontsize=12)
# fig.text(0.0, 0.56, "Return", va="center", rotation="vertical", fontsize=12)

handles, labels = axes[0, 0].get_legend_handles_labels()

axes[AXES["Swimmer"]].legend(
    handles=handles[4:],
    labels=labels[4:],
    loc="upper left",
    bbox_to_anchor=(0.1, 1.05),
    frameon=False,
)
axes[AXES["Walker2d"]].legend(
    handles=handles[:4],
    labels=labels[:4],
    loc="upper left",
    bbox_to_anchor=(0.2, 1.05),
    frameon=False,
)

# axes[AXES["InvertedPendulum"]].legend(
#     handles=handles,
#     labels=labels,
#     loc="upper left",
#     bbox_to_anchor=(0, 0.75),
#     labelspacing=0.4,
#     frameon=False,
# )


plt.tight_layout(pad=0.2)
# plt.savefig("adversarial_robust_standard.pdf")
# plt.savefig("adversarial_robust.pdf")

# plt.show()
