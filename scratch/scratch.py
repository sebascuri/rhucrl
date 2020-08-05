"""Python Script Template."""
from rhucrl.algorithm.adversarial_mpc import adversarial_solver
from rhucrl.environment.adversarial_environment import AdversarialEnv
from rhucrl.environment.wrappers import NoisyActionRobustWrapper
from rhucrl.environment.wrappers import RewardWrapper, ResetWrapper
from rllib.model.abstract_model import AbstractModel
from rllib.reward.abstract_reward import AbstractReward

from rllib.algorithms.mpc import CEMShooting, MPPIShooting

from rllib.policy.mpc_policy import MPCPolicy
from rllib.agent import MPCAgent
from rllib.util.training import train_agent, evaluate_agent
import torch
import numpy as np

ENV = "PendulumAdvEnv-v0"
SEED = 0
ALPHA = 0.1

np.random.seed(SEED)
torch.manual_seed(SEED)


def pendulum_reward(state, action, next_state, info):
    th, thdot = np.arctan2(state[..., 1], state[..., 0]), state[..., 2]
    action = action[..., 0]

    return -(th ** 2 + 0.1 * thdot ** 2 + 0.001 * (action ** 2))


def pendulum_reset(wrapper, **kwargs):
    high = np.array([np.pi, 0])
    wrapper.env.unwrapped.state = wrapper.env.np_random.uniform(low=high, high=high)
    wrapper.env.unwrapped.last_u = None
    return wrapper.env.unwrapped._get_obs()


env = AdversarialEnv(ENV, seed=SEED, alpha=ALPHA)
# env.add_wrapper(NoisyActionRobustWrapper, alpha=ALPHA)
env.add_wrapper(RewardWrapper, reward_function=pendulum_reward)
env.add_wrapper(ResetWrapper, reset_function=pendulum_reset)
print(env.adversarial_dim_action, env.protagonist_dim_action)
print(env.action_scale)


class PendulumReward(AbstractReward):
    def forward(self, state, action, next_state):
        th, thdot = torch.atan2(state[..., 1], state[..., 0]), state[..., 2]
        action = action[..., 0]

        return -(th ** 2 + 0.1 * thdot ** 2 + 0.001 * (action ** 2)), torch.tensor(0.0)


class PendulumModel(AbstractModel):
    def __init__(self, alpha):
        super().__init__(
            dim_state=(3,), dim_action=(2,),
        )
        self.max_speed = 8
        self.max_torque = 2.0
        self.alpha = alpha

    def forward(self, state, action):
        th, thdot = torch.atan2(state[..., 1], state[..., 0]), state[..., 2]

        p_action = action[..., 0]
        a_action = action[..., 1]
        # action = (1 - self.alpha) * p_action + self.alpha * a_action
        action = p_action
        g = 10.0
        m = 1.0 + a_action
        l = 1.0
        dt = 0.05

        u = torch.clamp(action, -self.max_torque, self.max_torque)

        newthdot = (
            thdot
            + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
        )
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -self.max_speed, self.max_speed)

        return (
            torch.stack((torch.cos(newth), torch.sin(newth), newthdot), dim=-1),
            torch.tensor(0.0),
        )


mpc_solver = adversarial_solver(
    base_solver=CEMShooting(
        dynamical_model=PendulumModel(alpha=ALPHA),
        reward_model=PendulumReward(),
        horizon=20,
        gamma=1.0,
        num_iter=5,
        num_samples=400,
        num_elites=40,
        termination=None,
        terminal_reward=None,
        warm_start=True,
        default_action="zero",
        num_cpu=1,
        action_scale=env.action_scale,
    ),
    protagonist_dim_action=env.protagonist_dim_action,
    adversarial_dim_action=env.adversarial_dim_action,
)

policy = MPCPolicy(mpc_solver)

agent = MPCAgent(mpc_policy=policy)

train_agent(
    environment=env,
    agent=agent,
    max_steps=200,
    num_episodes=10,
    render=True,
    print_frequency=1,
)
