"""Python Script Template."""
import matplotlib.pyplot as plt
import pandas as pd

from applications.plotters import AXES, COLORS, LABELS, LINESTYLE, set_figure_params

df = pd.read_json("action_robust.json")
group = df.groupby(["environment", "kind", "agent", "config", "alpha"])
envs = df.environment.unique()


def plot_action_rhucrl(
    ax,
    alpha,
    train_returns,
    environment,
    kind,
    config,
    beta=0.5,
    agent="RHUCRL",
    label=None,
):
    """Plot RHUCRL."""
    if label is None:
        label = agent
    ax.plot(
        alpha,
        train_returns.max()[environment][kind][agent][config],
        color=COLORS[label],
        label=LABELS[label],
        linestyle=LINESTYLE[label],
    )
    ax.fill_between(
        alpha,
        train_returns.max()[environment][kind][agent][config]
        - beta * train_returns.std()[environment][kind][agent][config],
        train_returns.max()[environment][kind][agent][config]
        + beta * train_returns.std()[environment][kind][agent][config],
        alpha=0.3,
        color=COLORS[label],
    )


def plot_action_robust(
    ax, alpha, train_returns, environment, kind, agent, config, beta=0.5, label=None
):
    """Plot algorithms."""
    if label is None:
        label = agent

    ax.plot(
        alpha,
        train_returns.mean()[environment][kind][agent][config],
        color=COLORS[label],
        label=LABELS[label],
        linestyle=LINESTYLE[label],
    )
    ax.fill_between(
        alpha,
        train_returns.mean()[environment][kind][agent][config]
        - beta * train_returns.std()[environment][kind][agent][config],
        train_returns.mean()[environment][kind][agent][config]
        + beta * train_returns.std()[environment][kind][agent][config],
        alpha=0.3,
        color=COLORS[label],
    )


set_figure_params(serif=True, fontsize=10)
fig, axes = plt.subplots(ncols=3, nrows=2, sharex="all")
fig.set_size_inches(6.5, 3.5)

kind = "noisy"
# kind = "probabilistic"
alpha = [0.1, 0.2, 0.3]

# %% Half Cheetah
env = "HalfCheetah"
ax = axes[AXES[env]]

plot_action_rhucrl(ax, alpha, group.robust_returns, env, kind, "hucrl_c")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "BestResponse", "hucrl_a"
)
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "MaxiMin", "hucrl_a")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "MaxiMin", "sac", label="MaxiMinMF"
)

plot_action_robust(ax, alpha, group.robust_returns, env, kind, "BPTT", "hucrl_c")
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "ActionRobust", "dpg")

ax.set_title(env)

# %% Hopper
env = "Hopper"
ax = axes[AXES[env]]

plot_action_rhucrl(ax, alpha, group.robust_returns, env, kind, "hucrl_a")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "BestResponse", "hucrl_c"
)
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "MaxiMin", "hucrl_c")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "MaxiMin", "sac", label="MaxiMinMF"
)

plot_action_robust(ax, alpha, group.robust_returns, env, kind, "BPTT", "hucrl_c")
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "ActionRobust", "dpg")
ax.set_title(env)


# %% Reacher2d
env = "Reacher2d"
ax = axes[AXES[env]]

plot_action_rhucrl(ax, alpha, group.robust_returns, env, kind, "hucrl_a")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "BestResponse", "hucrl_a"
)
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "MaxiMin", "hucrl_c")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "MaxiMin", "sac", label="MaxiMinMF"
)

plot_action_robust(ax, alpha, group.robust_returns, env, kind, "BPTT", "hucrl_c")
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "ActionRobust", "dpg")

ax.set_title(env)


# %% InvertedPendulum
env = "InvertedPendulum"
ax = axes[AXES[env]]

plot_action_rhucrl(ax, alpha, group.robust_returns, env, kind, "hucrl_b")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "BestResponse", "hucrl_c"
)
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "MaxiMin", "hucrl_c")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "MaxiMin", "sac", label="MaxiMinMF"
)

plot_action_robust(ax, alpha, group.robust_returns, env, kind, "BPTT", "hucrl_c")
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "ActionRobust", "dpg")

ax.set_title(env)


# %% Swimmer
env = "Swimmer"
ax = axes[AXES[env]]

plot_action_rhucrl(ax, alpha, group.robust_returns, env, kind, "hucrl_a")
# plot_action_robust(
#     ax, alpha, group.robust_returns, env, kind, "BestResponse", "hucrl_c"
# )
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "MaxiMin", "hucrl_c")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "MaxiMin", "sac", label="MaxiMinMF"
)

plot_action_robust(ax, alpha, group.robust_returns, env, kind, "BPTT", "hucrl_c")
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "ActionRobust", "dpg")

ax.set_title(env)


# %% Walker2d
env = "Walker2d"
ax = axes[AXES[env]]

plot_action_rhucrl(ax, alpha, group.robust_returns, env, kind, "hucrl_a")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "BestResponse", "hucrl_a"
)
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "MaxiMin", "hucrl_c")
plot_action_robust(
    ax, alpha, group.robust_returns, env, kind, "MaxiMin", "sac", label="MaxiMinMF"
)

plot_action_robust(ax, alpha, group.robust_returns, env, kind, "BPTT", "hucrl_c")
plot_action_robust(ax, alpha, group.robust_returns, env, kind, "ActionRobust", "dpg")

ax.set_title(env)


# %% Plot Formatting
axes[1, 1].set_xlabel("Mixture Parameter")
fig.text(0.0, 0.5, "Return", va="center", rotation="vertical", fontsize=12)

handles, labels = axes[0, 0].get_legend_handles_labels()
axes[AXES["Hopper"]].legend(
    handles=handles[:],
    labels=labels[:],
    loc="upper left",
    bbox_to_anchor=(0.01, 1.05),
    frameon=False,
)
# axes[1, 2].legend(
#     handles=handles[:4],
#     labels=labels[:4],
#     loc="upper left",
#     bbox_to_anchor=(0.0, 1.05),
#     frameon=False,
# )


plt.tight_layout(pad=0.2)
plt.savefig("action_robust.pdf")
# plt.show()
