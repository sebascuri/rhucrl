"""Adversarial RL environments."""

from gym.envs.registration import register

from .adversarial_environment import AdversarialEnv

ENVIRONMENTS = [
    "AntAdvEnv-v0",
    "HalfCheetahAdvEnv-v0",
    "HopperAdvEnv-v0",
    "HumanoidAdvEnv-v0",
    "HumanoidStandupAdvEnv-v0",
    "InvertedDoublePendulumAdvEnv-v0",
    "InvertedPendulumAdvEnv-v0",
    "Walker2dAdvEnv-v0",
    "ReacherAdvEnv-v0",
    "SwimmerAdvEnv-v0",
    "PusherAdvEnv-v0",
    "ThrowerAdvEnv-v0",
    "StrikerAdvEnv-v0",
]

# %% Adversarial MuJoCo
# ----------------------------------------

register(
    id="AntAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:get_ant_torso_env",
)

register(
    id="HalfCheetahAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:get_half_cheetah_torso_env",
)

register(
    id="HopperAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:get_hopper_torso_env",
)

register(
    id="HumanoidAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:get_humanoid_torso_env",
)

register(
    id="HumanoidStandupAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:"
    "get_humanoid_standup_torso_env",
)

register(
    id="InvertedDoublePendulumAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper"
    ":get_inverted_double_pendulum_pole2_env",
)

register(
    id="InvertedPendulumAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:"
    "get_inverted_pendulum_pole_env",
)


register(
    id="Walker2dAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:get_walker_torso_env",
)

register(
    id="ReacherAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:get_reacher_body1_env",
)

register(
    id="SwimmerAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:get_swimmer_env",
)

register(
    id="PusherAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:get_pusher_env",
)

register(
    id="ThrowerAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:get_thrower_env",
)


register(
    id="StrikerAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.mujoco_wrapper:get_stricker_env",
)


register(
    id="PendulumAdvEnv-v0",
    entry_point="rhucrl.environment.wrappers.pendulum_wrapper:PendulumAdvEnv",
)
