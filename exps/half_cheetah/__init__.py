"""Python Script Template."""
from gym.envs.registration import register

register(
    id="HalfCheetah-v4", entry_point="exps.half_cheetah.utilities:HalfCheetahV4Env"
)
