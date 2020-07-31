"""Python Script Template."""
from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.agent.rarl_agent import RARLAgent
from rhucrl.agent.zero_sum_agent import ZeroSumAgent
from rhucrl.policy.split_policy import SplitPolicy


from rllib.util.training import train_agent
from rllib.util.neural_networks.utilities import update_parameters

from gym.spaces import Box
from rhucrl.environment.adversarial_wrapper import (
    NoisyActionRobustWrapper,
    ProbabilisticActionRobustWrapper,
)

from rhucrl.environment import ENVIRONMENTS
from importlib import import_module


env_name = "HalfCheetahEnvAdv-v0"
base_agent_name = "SAC"
exploration_steps = 20000
max_steps = 1000
num_episodes = 200


def get_rarl_learner_agent(env_name, agent_name):
    """Learner agent."""
    learner_environment = AdversarialEnv(env_name)
    protagonist_dim_action = learner_environment.protagonist_dim_action[0]

    learner_environment.dim_action = (protagonist_dim_action,)
    learner_environment.action_space = Box(
        learner_environment.action_space.low[:protagonist_dim_action],
        learner_environment.action_space.high[:protagonist_dim_action],
    )

    module = import_module("rllib.agent")
    agent = getattr(module, f"{agent_name}Agent")

    learner_agent = agent.default(
        learner_environment, exploration_steps=exploration_steps
    )
    return learner_agent


def get_rarl_adversarial_agent(env_name, agent_name):
    """Adversarial agent."""
    adversarial_environment = AdversarialEnv(env_name)
    protagonist_dim_action = adversarial_environment.protagonist_dim_action[0]

    adversarial_environment.dim_action = (
        adversarial_environment.dim_action[0] - protagonist_dim_action,
    )
    adversarial_environment.action_space = Box(
        adversarial_environment.action_space.low[protagonist_dim_action:],
        adversarial_environment.action_space.high[protagonist_dim_action:],
    )

    module = import_module("rllib.agent")
    agent = getattr(module, f"{agent_name}Agent")

    adversarial_agent = agent.default(
        adversarial_environment, exploration_steps=exploration_steps
    )

    return adversarial_agent


def get_zero_sum_agents(env_name, agent_name):
    environment = AdversarialEnv(env_name)
    protagonist_dim_action = environment.protagonist_dim_action
    adversarial_dim_action = environment.adversarial_dim_action

    module = import_module("rllib.agent")
    agent = getattr(module, f"{agent_name}Agent")

    protagonist_agent = agent.default(environment, exploration_steps=exploration_steps)
    adversarial_agent = agent.default(environment, exploration_steps=exploration_steps)

    protagonist_policy = SplitPolicy(
        protagonist_agent.policy,
        protagonist_dim_action=protagonist_dim_action,
        adversarial_dim_action=adversarial_dim_action,
        protagonist=True,
    )
    adversarial_policy = SplitPolicy(
        protagonist_agent.policy,
        protagonist_dim_action=protagonist_dim_action,
        adversarial_dim_action=adversarial_dim_action,
        protagonist=False,
    )

    protagonist_agent.policy = protagonist_policy
    protagonist_agent.algorithm.policy = protagonist_policy

    adversarial_agent.policy = adversarial_policy
    adversarial_agent.algorithm.policy = adversarial_policy

    return protagonist_agent, adversarial_agent


rarl_agent = RARLAgent(
    get_rarl_learner_agent(env_name, base_agent_name),
    get_rarl_adversarial_agent(env_name, base_agent_name),
    exploration_steps=exploration_steps,
    train_frequency=50,
)
environment = AdversarialEnv(env_name,)
train_agent(
    rarl_agent,
    environment,
    max_steps=max_steps,
    num_episodes=num_episodes,
    print_frequency=1,
    render=False,
)
