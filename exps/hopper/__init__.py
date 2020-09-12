"""Python Script Template."""
from gym.envs.registration import register

register(id="Hopper-v4", entry_point="exps.hopper.utilities:HopperV4Env")
