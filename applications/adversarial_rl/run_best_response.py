"""Python Script Template."""
from importlib import import_module

from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from rllib.util.training.agent_training import train_agent

from rhucrl.agent.rhucrl_agent import get_rhucrl_agent_class
from rhucrl.environment import AdversarialEnv
from rhucrl.environment.wrappers import MujocoAdversarialWrapper

alpha = 2.0
hallucinate = True

name = "MBHalfCheetah-v0"
base_agent = "MVE"
n_adversaries = 3
num_episodes = 1000
max_steps = 1000

# Define environment
environment = AdversarialEnv(name)
environment.add_wrapper(
    MujocoAdversarialWrapper, alpha=alpha, force_body_names=["torso"]
)
environment.add_wrapper(HallucinationWrapper)


# Define agents
agent_module = import_module("rllib.agent")
agent_ = get_rhucrl_agent_class(getattr(agent_module, f"{base_agent}Agent"))
agent = agent_.default(environment, best_response=True)

# Train agent
train_agent(
    agent=agent,
    environment=environment,
    num_episodes=num_episodes,
    max_steps=max_steps,
    print_frequency=1,
)
