"""Python Script Template."""

import socket

ENVIRONMENTS = [
    "MBHalfCheetah-v0",
    "MBHopper-v0",
    "MBWalker2d-v0",
    "MBSwimmer-v0",
    # "MBCartPole-v0",
    # "MBInvertedPendulum-v0",
    # "MBInvertedDoublePendulum-v0",
    "PendulumSwingUp-v0",
    # "MBReacher2d-v0",
    "MBReacher3d-v0",
]

PER_LOGIN_ENVIRONMENT = {
    "greedy": ENVIRONMENTS[:4],
    "shallow": ENVIRONMENTS[4:8],
    "supermodularity": ENVIRONMENTS[8:],
    "lo-login-02": ENVIRONMENTS,
    "lo-login-01": ENVIRONMENTS,
}


def get_environment():
    """Get environment at different host-names."""
    return PER_LOGIN_ENVIRONMENT.get(socket.gethostname(), ENVIRONMENTS)
