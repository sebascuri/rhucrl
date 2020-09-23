"""Python Script Template."""

import socket

ENVIRONMENTS = {
    "greedy": ["MBHalfCheetah-v0", "MBHopper-v0", "MBWalker2d-v0", "MBSwimmer-v0"],
    "shallow": [
        "MBCartPole-v0",
        "MBInvertedPendulum-v0",
        "MBInvertedDoublePendulum-v0",
        "PendulumSwingUp-v0",
    ],
    "supermodularity": ["MBReacher2d-v0", "MBReacher3d-v0"],
}


def get_environment():
    """Get environment at different host-names."""
    return ENVIRONMENTS[socket.gethostname()]
