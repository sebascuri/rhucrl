"""Python Script Template."""
from importlib import import_module

from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from rllib.util.training.agent_training import train_agent

from rhucrl.agent import RARLAgent
from rhucrl.environment import AdversarialEnv
from rhucrl.environment.utilities import (
    adversarial_to_antagonist_environment,
    adversarial_to_protagonist_environment,
)
from rhucrl.environment.wrappers import MujocoAdversarialWrapper

alpha = 2.0
hallucinate_protagonist = True
hallucinate_antagonist = True


name = "MBHalfCheetah-v0"
base_agent = "SAC"
n_adversaries = 3
num_episodes = 1000
max_steps = 1000

# Define environment
environment = AdversarialEnv(name)
environment.add_wrapper(
    MujocoAdversarialWrapper, alpha=alpha, force_body_names=["torso"]
)
protagonist_env = adversarial_to_protagonist_environment(environment=environment)
antagonist_env = adversarial_to_antagonist_environment(environment=environment)

if hallucinate_protagonist:
    protagonist_env.add_wrapper(HallucinationWrapper)
if hallucinate_antagonist:
    antagonist_env.add_wrapper(HallucinationWrapper)

if hallucinate_protagonist or hallucinate_antagonist:
    environment.add_wrapper(HallucinationWrapper)

# Define agents
agent_module = import_module("rllib.agent")
agent_ = getattr(agent_module, f"{base_agent}Agent")
protagonist = agent_.default(protagonist_env)
antagonist = agent_.default(antagonist_env)

agent = RARLAgent(
    dim_action=environment.dim_action,
    action_scale=environment.action_scale,
    protagonist_agent=protagonist,
    antagonist_agent=antagonist,
    hallucinate=hallucinate_protagonist or hallucinate_antagonist,
)

# Train agent
train_agent(
    agent=agent,
    environment=environment,
    num_episodes=num_episodes,
    max_steps=max_steps,
    print_frequency=1,
)
