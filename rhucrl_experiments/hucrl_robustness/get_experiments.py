"""Python Script Template."""

import socket

ENVIRONMENTS = [
    "MBHalfCheetah-v0",
    "MBHopper-v0",
    "MBWalker2d-v0",
    "MBSwimmer-v0",
    "MBCartPole-v0",
    "MBInvertedPendulum-v0",
    "MBInvertedDoublePendulum-v0",
    "PendulumSwingUp-v0",
    "MBReacher2d-v0",
    "MBReacher3d-v0",
]


PER_LOGIN_ENVIRONMENT = {
    "greedy": ENVIRONMENTS[:2],
    "shallow": ENVIRONMENTS[2:4],
    "supermodularity": ENVIRONMENTS[4:],
}


def get_environment():
    """Get environment at different host-names."""
    return PER_LOGIN_ENVIRONMENT.get(socket.gethostname(), ENVIRONMENTS)


def get_experiments():
    """Get experiments."""
    mujoco_envs = get_environment().copy()
    if "PendulumSwingUp-v0" in mujoco_envs:
        mujoco_envs.remove("PendulumSwingUp-v0")

    experiments = [
        {
            "environment": get_environment(),
            "alpha": [0.01, 0.05, 0.1, 0.15, 0.2],
            "wrapper": ["noisy_action", "probabilistic_action"],
        },
        {
            "environment": mujoco_envs,
            "alpha": [1.0, 5.0, 10.0],
            "wrapper": ["external_force"],
        },
        {
            "environment": ["PendulumSwingUp-v0"],
            "alpha": [0.01, 0.05, 0.1, 0.15, 0.2],
            "wrapper": ["adversarial_pendulum"],
        },
    ]
    return experiments
