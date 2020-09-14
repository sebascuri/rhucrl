from gym.envs.registration import register

register(
    id="Pendulum-v1",
    entry_point="exps.inverted_pendulum_swingup.utilities:PendulumV1Env",
)
