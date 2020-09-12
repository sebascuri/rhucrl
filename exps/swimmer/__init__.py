"""Python Script Template."""
from gym.envs.registration import register

register(id="Swimmer-v4", entry_point="exps.swimmer.utilities:SwimmerV4Env")
