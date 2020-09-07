"""Python Script Template."""
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
from rllib.agent import AbstractAgent
from rllib.environment import AbstractEnvironment
from rllib.util.logger import Logger
from rllib.util.rollout import rollout_agent

from rhucrl.agent.adversarial_agent import AdversarialAgent


def plot_logger(logger: Logger, title_str: str) -> None:
    """Plott logger."""
    for key in logger.keys:
        plt.plot(logger.get(key))
        plt.xlabel("Episode")
        plt.ylabel(" ".join(key.split("_")).title())
        plt.title(title_str)
        plt.show()


def train_adversarial_agent(
    agent: AdversarialAgent,
    environment: AbstractEnvironment,
    num_episodes: int,
    max_steps: int,
    mode: str = "both",
    plot_flag: bool = True,
    print_frequency: int = 0,
    eval_frequency: int = 0,
    plot_frequency: int = 0,
    save_milestones: Optional[List[int]] = None,
    render: bool = False,
    plot_callbacks: Optional[List[Callable[[AbstractAgent, int], None]]] = None,
) -> None:
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
