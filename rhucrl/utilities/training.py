"""Python Script Template."""
import matplotlib.pyplot as plt
from rllib.util.rollout import rollout_agent

from rhucrl.agent.adversarial_agent import AdversarialAgent


def plot_logger(logger, title_str):
    """Plott logger."""
    for key in logger.keys:
        plt.plot(logger.get(key))
        plt.xlabel("Episode")
        plt.ylabel(" ".join(key.split("_")).title())
        plt.title(title_str)
        plt.show()


def train_adversarial_agent(
    agent: AdversarialAgent,
    environment,
    num_episodes,
    max_steps,
    mode="both",
    plot_flag=True,
    print_frequency=0,
    eval_frequency=0,
    plot_frequency=0,
    save_milestones=None,
    render=False,
    plot_callbacks=None,
):
    """Train an agent in an environment.

    Parameters
    ----------
    agent: AdversarialAgent
    environment: AdversarialEnvironment
    num_episodes: int
    max_steps: int
    mode: str.
        Mode to train an adversarial agent.
    plot_flag: bool, optional.
    print_frequency: int, optional.
    eval_frequency: int, optional.
    plot_frequency: int
    save_milestones: List[int], optional.
        List with episodes in which to save the agent.
    render: bool, optional.
    plot_callbacks: list, optional.

    """
    allowed = ["both", "protagonist", "antagonist"]
    if mode == "both":
        agent.train()
    elif mode == "protagonist":
        agent.train_only_protagonist()
    elif mode == "antagonist":
        agent.train_only_antagonist()
    else:
        raise NotImplementedError(f"{mode} has to be in {allowed}")

    rollout_agent(
        environment,
        agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
        print_frequency=print_frequency,
        plot_frequency=plot_frequency,
        eval_frequency=eval_frequency,
        save_milestones=save_milestones,
        render=render,
        plot_callbacks=plot_callbacks,
    )

    if plot_flag:
        plot_logger(agent.logger, f"{agent.name} in {environment.name}")
        plot_logger(
            agent.protagonist_agent.logger, f"Protagonist in {environment.name}"
        )
        plot_logger(agent.antagonist_agent.logger, f"Antagonist in {environment.name}")
    print(agent)
