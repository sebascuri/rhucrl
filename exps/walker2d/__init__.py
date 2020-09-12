"""Python Script Template."""
from gym.envs.registration import register

register(id="Walker2d-v4", entry_point="exps.walker2d.utilities:Walker2dV4Env")
