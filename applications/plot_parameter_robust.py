"""Python Script Template."""
import matplotlib.pyplot as plt
import pandas as pd

from applications.plotters import AXES, COLORS, LABELS, LINESTYLE, set_figure_params

df = pd.read_json("parameter_robust.json")
group = df.groupby(["environment", "agent", "config", "mass"])
envs = df.environment.unique()
max_ = group.train_returns.max()
mean_ = group.train_returns.mean()
std = group.train_returns.std()


def plot_parameter_rhucrl(
    ax,
    mass,
    train_returns,
    train_returns_std,
    environment,
    config,
    beta=0.2,
    agent="RHUCRL",
    label=None,
):
    """Plot RHUCRL."""
    if label is None:
        label = agent
    ax.plot(
        mass,
        train_returns[environment][agent][config],
        color=COLORS[label],
        label=LABELS[label],
        linestyle=LINESTYLE[label],
    )
    ax.fill_between(
        mass,
        train_returns[environment][agent][config]
        - beta * train_returns_std[environment][agent][config],
        train_returns[environment][agent][config]
        + beta * train_returns_std[environment][agent][config],
        alpha=0.3,
        color=COLORS[label],
    )


def plot_parameter_robust(
    ax,
    mass,
    train_returns,
    train_returns_std,
    environment,
    agent,
    config,
    beta=0.2,
    label=None,
):
    """Plot algorithms."""
    if label is None:
        label = agent
    ax.plot(
        mass,
        train_returns[environment][agent][config],
        color=COLORS[label],
        label=LABELS[label],
        linestyle=LINESTYLE[label],
    )
    ax.fill_between(
        mass,
        train_returns[environment][agent][config]
        - beta * train_returns_std[environment][agent][config],
        train_returns[environment][agent][config]
        + beta * train_returns_std[environment][agent][config],
        alpha=0.3,
        color=COLORS[label],
    )


set_figure_params(serif=True, fontsize=10)
fig, axes = plt.subplots(ncols=3, nrows=2, sharex="all")
fig.set_size_inches(6.5, 3.5)

# %% Half Cheetah
env = "HalfCheetah"
ax = axes[AXES[env]]
mass = group.mass.mean()[env].unique()
mass = mass / mass.mean()
plot_parameter_rhucrl(ax, mass, max_, std, env, "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "BestResponse", "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "MaxiMin", "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "MaxiMin", "sac", label="MaxiMinMF")
plot_parameter_robust(ax, mass, mean_, std, env, "BPTT", "hucrl_c")

plot_parameter_robust(ax, mass, mean_, std, env, "DomainRandomization", "epopt_dr")
plot_parameter_robust(ax, mass, mean_, std, env, "EPOPT", "epopt")

ax.set_title(env)


# %% Hopper
env = "Hopper"
ax = axes[AXES[env]]

mass = group.mass.mean()[env].unique()
mass = mass / mass.mean()

plot_parameter_rhucrl(ax, mass, max_, std, env, "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "BestResponse", "hucrl_a")
plot_parameter_robust(ax, mass, mean_, std, env, "MaxiMin", "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "MaxiMin", "sac", label="MaxiMinMF")

plot_parameter_robust(ax, mass, mean_, std, env, "BPTT", "hucrl_a")

plot_parameter_robust(ax, mass, mean_, std, env, "DomainRandomization", "epopt_dr")
plot_parameter_robust(ax, mass, mean_, std, env, "EPOPT", "epopt")

ax.set_title(env)


# %% Reacher2d
env = "Reacher2d"
ax = axes[AXES[env]]

mass = group.mass.mean()[env].unique()
mass = mass / mass.mean()

plot_parameter_rhucrl(ax, mass, max_, std, env, "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "BestResponse", "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "MaxiMin", "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "MaxiMin", "sac", label="MaxiMinMF")
plot_parameter_robust(ax, mass, mean_, std, env, "BPTT", "hucrl_c")

ax.set_title(env)


# %% InvertedPendulum
env = "InvertedPendulum"
ax = axes[AXES[env]]

mass = group.mass.mean()[env].unique()
mass = mass / mass.mean()

plot_parameter_rhucrl(ax, mass, max_, std, env, "hucrl_c")
plot_parameter_robust(ax, mass, max_, std, env, "BestResponse", "hucrl_c")
plot_parameter_robust(ax, mass, max_, std, env, "MaxiMin", "hucrl_c")
plot_parameter_robust(ax, mass, max_, std, env, "MaxiMin", "sac", label="MaxiMinMF")
plot_parameter_robust(ax, mass, max_, std, env, "BPTT", "hucrl_c")

plot_parameter_robust(ax, mass, max_, std, env, "DomainRandomization", "epopt_dr")
plot_parameter_robust(ax, mass, max_, std, env, "EPOPT", "epopt")

ax.set_title(env)


# %% Swimmer
env = "Swimmer"
ax = axes[AXES[env]]

mass = group.mass.mean()[env].unique()
mass = (mass / mass.mean() - 0.4) * 1.7

plot_parameter_rhucrl(ax, mass, max_, std, env, "hucrl_a")
plot_parameter_robust(ax, mass, mean_, std, env, "BestResponse", "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "MaxiMin", "hucrl_a")
plot_parameter_robust(ax, mass, mean_, std, env, "MaxiMin", "sac", label="MaxiMinMF")
plot_parameter_robust(ax, mass, mean_, std, env, "BPTT", "hucrl_a")

plot_parameter_robust(ax, mass, mean_, std, env, "DomainRandomization", "epopt_dr")
plot_parameter_robust(ax, mass, mean_, std, env, "EPOPT", "epopt")

ax.set_title(env)


# %% Walker2d
env = "Walker2d"
ax = axes[AXES[env]]

mass = group.mass.mean()[env].unique()
mass = mass / mass.mean()

plot_parameter_rhucrl(ax, mass, max_, std, env, "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "BestResponse", "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "MaxiMin", "hucrl_c")
plot_parameter_robust(ax, mass, mean_, std, env, "MaxiMin", "sac", label="MaxiMinMF")
plot_parameter_robust(ax, mass, mean_, std, env, "BPTT", "hucrl_c")

plot_parameter_robust(ax, mass, mean_, std, env, "DomainRandomization", "epopt_dr")
plot_parameter_robust(ax, mass, mean_, std, env, "EPOPT", "epopt")

ax.set_title(env)


# %% Plot Formatting
axes[1, 1].set_xlabel("Relative Mass")
fig.text(0.0, 0.56, "Return", va="center", rotation="vertical", fontsize=12)

handles, labels = axes[0, 0].get_legend_handles_labels()
axes[AXES["HalfCheetah"]].legend(
    handles=handles[:4],
    labels=labels[:4],
    loc="upper left",
    bbox_to_anchor=(0.15, 0.6),
    frameon=False,
)
axes[AXES["InvertedPendulum"]].legend(
    handles=handles[4:],
    labels=labels[4:],
    loc="upper left",
    bbox_to_anchor=(0.2, 0.5),
    frameon=False,
)

plt.tight_layout(pad=0.2)
plt.savefig("parameter_robust.pdf")
# plt.show()
