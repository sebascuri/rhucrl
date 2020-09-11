from gym.envs.registration import register

register(id="Pendulum-v1", entry_point="exps.pendulum.utilities:PendulumV1Env")
