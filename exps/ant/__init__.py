"""Python Script Template."""
from gym.envs.registration import register

register(id="Ant-v4", entry_point="exps.ant.utilities:AntV4Env")
