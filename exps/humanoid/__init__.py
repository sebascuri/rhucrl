"""Python Script Template."""
from gym.envs.registration import register

register(id="Humanoid-v4", entry_point="exps.humanoid.utilities:HumanoidV4Env")
