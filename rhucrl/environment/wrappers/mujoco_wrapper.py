"""Python Script Template."""

from typing import Dict, List, Optional, Tuple

import gym.error
import numpy as np

from .adversarial_wrapper import AdversarialWrapper

try:
    from gym.envs.mujoco import (
        AntEnv,
        HalfCheetahEnv,
        HopperEnv,
        HumanoidEnv,
        HumanoidStandupEnv,
        InvertedDoublePendulumEnv,
        InvertedPendulumEnv,
        MujocoEnv,
        PusherEnv,
        ReacherEnv,
        StrikerEnv,
        SwimmerEnv,
        ThrowerEnv,
        Walker2dEnv,
    )

    class MujocoAdversarialWrapper(AdversarialWrapper):
        """Wrapper for Mujoco adversarial environments."""

        def __init__(
            self,
            env: MujocoEnv,
            alpha: float = 5.0,
            force_body_names: Optional[List[str]] = None,
            new_mass: Optional[Dict[str, float]] = None,
            new_friction: Optional[Dict[str, float]] = None,
        ):
            force_body_names = [] if force_body_names is None else force_body_names
            self._antagonist_bindex = [
                env.model.body_names.index(i) for i in force_body_names
            ]
            antagonist_high = np.ones(2 * len(force_body_names))
            antagonist_low = -antagonist_high

            new_mass = {} if new_mass is None else new_mass
            for body_name, weight in new_mass.items():
                env.model.body_mass[env.model.body_names.index(body_name)] = weight
            new_friction = {} if new_friction is None else new_friction
            for body_name, friction in new_friction.items():
                env.model.geom_friction[
                    env.model.body_names.index(body_name), 0
                ] = friction

            super().__init__(
                env=env,
                antagonist_low=antagonist_low,
                antagonist_high=antagonist_high,
                alpha=alpha,
            )

        def _antagonist_action_to_xfrc(self, antagonist_action: np.ndarray) -> None:
            for i, bindex in enumerate(self._antagonist_bindex):
                self.sim.data.xfrc_applied[bindex] = np.array(
                    [
                        antagonist_action[i * 2],
                        0.0,
                        antagonist_action[i * 2 + 1],
                        0.0,
                        0.0,
                        0.0,
                    ]
                )

        def adversarial_step(
            self, protagonist_action: np.ndarray, antagonist_action: np.ndarray
        ) -> Tuple[np.ndarray, float, bool, dict]:
            """See `AdversarialWrapper.adversarial_step()'."""
            self._antagonist_action_to_xfrc(antagonist_action)
            return self.env.step(protagonist_action)

    def get_ant_torso_env(**kwargs):
        """Get AntEnv with perturbations in the torso and heel."""
        return MujocoAdversarialWrapper(
            env=AntEnv(), force_body_names=["torso"], **kwargs
        )

    def get_half_cheetah_torso_env(**kwargs):
        """Get HalfCheetahEnv with perturbations in the torso."""
        return MujocoAdversarialWrapper(
            env=HalfCheetahEnv(), force_body_names=["torso"], **kwargs
        )

    def get_hopper_torso_env(**kwargs):
        """Get HopperEnv with perturbations in the torso and heel."""
        return MujocoAdversarialWrapper(
            env=HopperEnv(), force_body_names=["torso"], **kwargs
        )

    def get_humanoid_torso_env(**kwargs):
        """Get HumanoidEnv with perturbations in the torso and heel."""
        return MujocoAdversarialWrapper(
            env=HumanoidEnv(), force_body_names=["torso"], **kwargs
        )

    def get_humanoid_standup_torso_env(**kwargs):
        """Get HumanoidStandupEnv with perturbations in the torso and heel."""
        return MujocoAdversarialWrapper(
            env=HumanoidStandupEnv(), force_body_names=["torso"], **kwargs
        )

    def get_inverted_pendulum_pole_env(**kwargs):
        """Get InvertedPendulumEnv with perturbations in the pole."""
        return MujocoAdversarialWrapper(
            env=InvertedPendulumEnv(), force_body_names=["pole"], **kwargs
        )

    def get_inverted_double_pendulum_pole2_env(**kwargs):
        """Get InvertedDoublePendulumEnv with perturbations in the second pole."""
        return MujocoAdversarialWrapper(
            env=InvertedDoublePendulumEnv(), force_body_names=["pole2"], **kwargs
        )

    def get_walker_torso_env(**kwargs):
        """Get Walker2dEnv with perturbations in the pole."""
        return MujocoAdversarialWrapper(
            env=Walker2dEnv(), force_body_names=["torso"], **kwargs
        )

    def get_reacher_body1_env(**kwargs):
        """Get ReacherEnv with perturbations in the body1 link."""
        return MujocoAdversarialWrapper(
            env=ReacherEnv(), force_body_names=["body1"], **kwargs
        )

    def get_swimmer_env(**kwargs):
        """Get SwimmerEnv with perturbations in the torso."""
        return MujocoAdversarialWrapper(
            env=SwimmerEnv(), force_body_names=["torso"], **kwargs
        )

    def get_pusher_env(**kwargs):
        """Get PusherEnv with perturbations in the r_shoulder_lift_link."""
        return MujocoAdversarialWrapper(
            env=PusherEnv(), force_body_names=["r_shoulder_lift_link"], **kwargs
        )

    def get_thrower_env(**kwargs):
        """Get ThrowerEnv with perturbations in the r_shoulder_lift_link."""
        return MujocoAdversarialWrapper(
            env=ThrowerEnv(), force_body_names=["r_shoulder_lift_link"], **kwargs
        )

    def get_stricker_env(**kwargs):
        """Get StrikerEnv with perturbations in the r_shoulder_lift_link."""
        return MujocoAdversarialWrapper(
            env=StrikerEnv(), force_body_names=["r_shoulder_lift_link"], **kwargs
        )

    def get_half_cheetah_heel_env(**kwargs):
        """Get HalfCheetahEnv with perturbations in the heels."""
        return MujocoAdversarialWrapper(
            env=HalfCheetahEnv(), force_body_names=["bfoot", "ffoot"], **kwargs
        )

    def get_half_cheetah_torso_heel_env(**kwargs):
        """Get HalfCheetahEnv with perturbations in the torso and heel."""
        return MujocoAdversarialWrapper(
            env=HalfCheetahEnv(), force_body_names=["torso", "bfoot", "ffoot"], **kwargs
        )


except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    pass
