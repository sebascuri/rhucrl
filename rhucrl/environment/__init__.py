"""Adversarial RL environments."""

from gym.envs.registration import register

ENVIRONMENTS = [
    "AntEnvAdv-v0",
    "HalfCheetahEnvAdv-v0",
    "HopperEnvAdv-v0",
    "HumanoidEnvAdv-v0",
    "HumanoidStandupEnvAdv-v0",
    "InvertedDoublePendulumEnvAdv-v0",
    "InvertedPendulumEnvAdv-v0",
    "Walker2dEnvAdv-v0",
    "ReacherEnvAdv-v0",
    "SwimmerEnvAdv-v0",
    "PusherEnvAdv-v0",
    "ThrowerEnvAdv-v0",
    "StrikerEnvAdv-v0",
]

# %% Adversarial MuJoCo
# ----------------------------------------

register(
    id="AntEnvAdv-v0", entry_point="rhucrl.environment.mujoco_wrapper:get_ant_torso_env"
)

register(
    id="HalfCheetahEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper:get_half_cheetah_torso_env",
)

register(
    id="HopperEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper:get_hopper_torso_env",
)

register(
    id="HumanoidEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper:get_humanoid_torso_env",
)

register(
    id="HumanoidStandupEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper:get_humanoid_standup_torso_env",
)

register(
    id="InvertedDoublePendulumEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper"
    ":get_inverted_double_pendulum_pole2_env",
)

register(
    id="InvertedPendulumEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper:get_inverted_pendulum_pole_env",
)


register(
    id="Walker2dEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper:get_walker_torso_env",
)

register(
    id="ReacherEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper:get_reacher_body1_env",
)

register(
    id="SwimmerEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper:get_swimmer_env",
)

register(
    id="PusherEnvAdv-v0", entry_point="rhucrl.environment.mujoco_wrapper:get_pusher_env"
)

register(
    id="ThrowerEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper:get_thrower_env",
)


register(
    id="StrikerEnvAdv-v0",
    entry_point="rhucrl.environment.mujoco_wrapper:get_stricker_env",
)


register(
    id="PendulumAdv-v0",
    entry_point="rhucrl.environment.classic_control:AdversarialPendulumEnv",
)
