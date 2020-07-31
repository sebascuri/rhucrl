"""Python Script Template."""
from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.agent.zero_sum_agent import ZeroSumAgent
from rhucrl.policy.split_policy import SplitPolicy

from rllib.util.training import train_agent
from importlib import import_module


env_name = "HalfCheetahEnvAdv-v0"
base_agent_name = "SAC"
exploration_steps = 20000
max_steps = 1000
num_episodes = 200


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


rarl_agent = ZeroSumAgent(
    *get_zero_sum_agents(env_name, base_agent_name),
    exploration_steps=0,  # exploration_steps,
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
