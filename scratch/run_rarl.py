"""Python Script Template."""
from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.agent.rarl_agent import RARLAgent as Agent

from rllib.util.training.agent_training import train_agent
from rllib.util.utilities import set_random_seed


env_name = "HalfCheetahAdvEnv-v0"
base_agent_name = "SAC"
seed = 0
exploration_steps = 10000
max_steps = 1000
num_episodes = 200

set_random_seed(seed)

environment = AdversarialEnv(env_name, seed=seed)
agent = Agent.default(
    environment, base_agent_name=base_agent_name, exploration_steps=exploration_steps
)
train_agent(
    agent, environment, num_episodes, max_steps, print_frequency=1,
)
