"""Python Script Template."""

import socket

ENVIRONMENTS = {
    "greedy": ["MBHalfCheetah-v0", "MBHopper-v0", "MBSwimmer-v0", "MBWalker2d-v0"],
    "shallow": [
        "PendulumSwingUp-v0",
        "MBCartPole-v0",
        "PendulumSwingUp-v0",
        "MBInvertedDoublePendulum-v0",
    ],
    "supermodularity": ["MBReacher2d-v0", "MBReacher3d-v0"],
}


def get_environment():
    """Get environment at different host-names."""
    return ENVIRONMENTS[socket.gethostname()]
